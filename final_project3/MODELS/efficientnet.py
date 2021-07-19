import torch
from torch import nn
from torch.nn import functional as F

from .efficientnet_utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv3d,
    get_model_params,
    efficientnet_params,
    Swish,
    MemoryEfficientSwish,
)

class MBConvBlock3D(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv3d = get_same_padding_conv3d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv3d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm3d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv3d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm3d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv3d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv3d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv3d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm3d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool3d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet3D(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet3D.from_pretrained('efficientnet-b0')

    """

    def __init__(self, blocks_args=None, global_params=None, in_channels=3, mode = 'single', class_num=4, label_num=4, vocab_size = 0):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args
        self.mode = mode
        self.embed_dim = 128
        self.audio_size = 1024
        self.label_num = label_num
        self.num_classes = class_num
        # Get static or dynamic convolution depending on image size
        Conv3d = get_same_padding_conv3d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv3d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm3d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock3D(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock3D(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm3d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool3d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._swish = MemoryEfficientSwish()


        self.classifier1 = nn.ModuleList([nn.Linear((256*32)+(self.embed_dim*2)+self.audio_size, self.num_classes) for _ in range(self.label_num)])
        
# text #
        self.embedding = nn.Embedding(vocab_size, self.embed_dim, padding_idx=0)
        self.textdp = nn.Dropout(0.4)
        #self.textconv = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1)
        self.textcnn = nn.Sequential(
                    nn.Conv1d(in_channels = 60, out_channels = self.embed_dim, kernel_size = 7, padding = 3),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Conv1d(in_channels = self.embed_dim, out_channels = self.embed_dim, kernel_size = 7, padding = 3),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Conv1d(in_channels = self.embed_dim, out_channels = self.embed_dim*2, kernel_size = 3, padding = 1),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Conv1d(in_channels = self.embed_dim*2, out_channels = self.embed_dim*2, kernel_size = 3, padding = 1),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Conv1d(in_channels = self.embed_dim*2, out_channels = self.embed_dim*2, kernel_size = 3, padding = 1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1)
                    )
        self.text_init_weights()

        # genre fc, age fc #
        self.genrefc = nn.Linear((256*32) + (self.embed_dim*2) + self.audio_size, 9)
        self.agefc = nn.Linear((256*32) + (self.embed_dim*2) + self.audio_size, 4)


        self.extract_audio = nn.Sequential(nn.Conv2d(1, 32, kernel_size=(3,15), stride=(1,3), padding=(1,1)),\
                                        nn.BatchNorm2d(32),\
                                        nn.LeakyReLU(),\
                                        nn.MaxPool2d(2),\
                                        nn.Conv2d(32, 64, kernel_size=(3,15), stride=(1,3), padding=(1,1)),\
                                        nn.BatchNorm2d(64),\
                                        nn.LeakyReLU(),\
                                        nn.MaxPool2d(2),\
                                        nn.Conv2d(64, 128, kernel_size=(3,15), stride=(1,3), padding=(1,1)),\
                                        nn.BatchNorm2d(128),\
                                        nn.LeakyReLU(),\
                                        nn.MaxPool2d(2),\
                                        nn.Conv2d(128, 256, kernel_size=(3,11), stride=(1,3), padding=(1,1)),\
                                        nn.AdaptiveAvgPool2d(2)\
                                        )

        self.lstm = nn.LSTM(1280, hidden_size = 256, num_layers = 2, batch_first=False, bidirectional=True, dropout=0.3) 


    def text_init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        #self.textfc1.weight.data.uniform_(-initrange, initrange)
        #self.textfc1.bias.data.zero_()
        #self.textfc2.weight.data.uniform_(-initrange, initrange)
        #self.textfc2.bias.data.zero_()

    def init_hidden(self,batch_size,device):
        hidden = (
        torch.zeros(4,batch_size,256).requires_grad_().to(device),
        torch.zeros(4,batch_size,256).requires_grad_().to(device),
        ) # num_layers, Batch, hidden size
        return hidden


    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)


    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs, text, audio):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)
        video = []
        for i in range(0, inputs.size(2), 64):
            # Convolution layers
            x = self.extract_features(inputs[:,:,i:i+64:1,:,:])

            # Pooling and final linear layer
            x = self._avg_pooling(x)
            x = x.view(bs, -1)
            video.append(x)
        video = torch.stack(video)
        hidden = self.init_hidden(bs, inputs.device)
        video, _ = self.lstm(video, hidden)
        video = video.view(video.size(1),-1)

        text = self.embedding(text)
        text = self.textdp(text)
        text = self.textcnn(text)
        text = text.view(text.size(0), -1)
        
        audio = self.extract_audio(audio)
        audio = audio.view(audio.shape[0], -1)
        features = torch.cat([video, text, audio], dim=1)
        features = self._dropout(features)
        # method 1
        outputs = []
        for fc in self.classifier1:
            outputs.append(fc(features))
        outputs = torch.stack(outputs)

        genre = self.genrefc(features)
        age = self.agefc(features)
        return outputs, genre, age

    @classmethod
    def from_name(cls, model_name, override_params=None, in_channels=3, mode='multi', class_num = 4, label_num=4, vocab_size = 0):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params, in_channels, mode, class_num, label_num, vocab_size)

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """ 
        valid_models = ['efficientnet-b'+str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))


def c2_msra_fill(module: nn.Module) -> None:
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity='relu')
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)

def init_weights(model, fc_init_std=0.01, zero_init_final_bn=True):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            c2_msra_fill(m)
        elif isinstance(m, nn.BatchNorm3d):
            batchnorm_weight=1.0
            if m.weight is not None:
                m.weight.data.fill_(batchnorm_weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std = fc_init_std)
            if m.bias is not None:
                m.bias.data.zero_()

if __name__=='__main__':

    model = EfficientNet3D.from_name('efficientnet-b0', override_params={'num_classes':2}, in_channels=3, mode='multi', label_num=4)
    model = model.cuda()
    inputs = torch.randn((8,3,256,112,112)).cuda()

    for _ in range(100):
        out = model(inputs)
        print(out.shape)
