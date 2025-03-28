#
# Swin Unet running with Tensorflow 2.16
# Its currently set up for binary and 1 class
# can be modified to more classes
#
#

import tensorflow as tf
from tensorflow.keras import layers, models
from keras_unet_collection.transformer_layers import patch_merging, patch_expanding
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout


# Custom PatchMerging and PatchExpanding classes
class CustomPatchMerging(Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.linear = Dense(self.embed_dim * 2, use_bias=False)

    def call(self, x):
        _, h, w, c = x.shape
        x = tf.reshape(x, (-1, h // 2, 2, w // 2, 2, c))  # Unterteile in 2x2 Patches
        x = tf.reduce_mean(x, axis=(2, 4))  # Berechne den Durchschnitt pro Patch
        x = self.linear(x)  # Wende lineare Projektion an
        return x


class CustomPatchExpanding(Layer):
    def __init__(self, embed_dim, upsample_rate=2, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.upsample_rate = upsample_rate

    def build(self, input_shape):
        self.linear = Dense(self.embed_dim, use_bias=False)

    def call(self, x):
        B, H, W, C = x.shape
        x = tf.image.resize(x, (H * self.upsample_rate, W * self.upsample_rate))  # Upsampling
        x = self.linear(x)  # Lineare Projektion auf neue Dimension
        return x


class PatchExtract(layers.Layer):
    def __init__(self, patch_size):
        super(PatchExtract, self).__init__()
        self.patch_size = patch_size

    def call(self, inputs):
        shape = tf.shape(inputs)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]

        patch_size = self.patch_size
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, patch_size[0], patch_size[1], 1],
            strides=[1, patch_size[0], patch_size[1], 1],
            rates=[1, 1, 1, 1],
            padding='SAME')
        return patches


class PatchEmbedding(layers.Layer):
    def __init__(self, num_patches, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.dense = layers.Dense(embed_dim)

    def call(self, inputs):
        return self.dense(inputs)


class WindowMultiHeadSelfAttention(Layer):
    def __init__(self, dim, num_heads, window_size, qkv_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim  
        self.num_heads = num_heads  
        self.window_size = window_size  
        self.scale = (dim // num_heads) ** -0.5  

        self.qkv = Dense(dim * 3, use_bias=qkv_bias)
        self.proj = Dense(dim)

    def call(self, x):
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], self.dim  
        x = tf.reshape(x, (B, H * W, C))  

        # Q, K, V aufteilen
        qkv = self.qkv(x)  
        qkv = tf.reshape(qkv, (B, H * W, 3, self.num_heads, C // self.num_heads))
        q, k, v = tf.unstack(qkv, axis=2)

        # Skaliertes Dot-Produkt Self Attention
        attn_scores = tf.matmul(q, k, transpose_b=True) * self.scale
        attn_weights = tf.nn.softmax(attn_scores, axis=-1)

        # Anwenden auf Werte
        attn_output = tf.matmul(attn_weights, v)

        # Rückprojektieren
        attn_output = tf.reshape(attn_output, (B, H * W, C))
        attn_output = self.proj(attn_output)

        return attn_output



class SwinTransformerBlock(Layer):
    def __init__(self, dim, num_patch, num_heads, window_size, shift_size, num_mlp, qkv_bias=True,qk_scale=None, mlp_drop=None,attn_drop=None, proj_drop=None, drop_path_prob=None, drop_rate=0.0, **kwargs):
        super(SwinTransformerBlock, self).__init__(**kwargs)
        self.dim = dim  
        self.num_patch = num_patch  
        self.num_heads = num_heads  
        self.window_size = window_size  
        self.shift_size = shift_size  
        self.num_mlp = num_mlp  
        self.qkv_bias = qkv_bias  
        self.qk_scale = qk_scale  
        self.drop_rate = drop_rate  
        self.mlp_drop = mlp_drop
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.drop_path_prob = drop_path_prob

        # Layer Normalization
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)

        # MSA mit Fensterung
        self.attn = WindowMultiHeadSelfAttention(dim, num_heads, window_size, qkv_bias)

        # MLP Feedforward
        self.mlp = tf.keras.Sequential([
            Dense(num_mlp * dim, activation="gelu"),  
            Dropout(drop_rate),
            Dense(dim),  
            Dropout(drop_rate)
        ])

    def call(self, inputs):
        shortcut = inputs  # Original Input Shape: (B, H, W, C)
        B, H, W, C = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]
        
        # LayerNorm + Multi-Head Attention (Self Attention)
        x = self.norm1(inputs)
        x = self.attn(x)  # Hier wird der Multi-Head Attention Layer angewendet
        x = tf.reshape(x, tf.shape(shortcut)) 
        # Residual Connection (shortcut + x)
        x = shortcut + x  # Residual Connection

        # LayerNorm + MLP Feedforward
        x = self.norm2(x)
        x = self.mlp(x)

        assert x.shape == shortcut.shape, f"Formen stimmen nicht überein: x={x.shape}, shortcut={shortcut.shape}"

        x = shortcut + x  # Residual Connection
        return x




def swin_transformer_stack(X, stack_num, embed_dim, num_patch, num_heads, window_size, num_mlp, shift_window=True, name=''):
    mlp_drop_rate = 0
    attn_drop_rate = 0
    proj_drop_rate = 0
    drop_path_rate = 0
    qkv_bias = True
    qk_scale = None
    
    if shift_window:
        shift_size = window_size // 2
    else:
        shift_size = 0
    
    for i in range(stack_num):
        shift_size_temp = 0 if i % 2 == 0 else shift_size

        # Dynamischer Name für Swin-Transformer-Blöcke
        block_name = '{}_swin_block_{}'.format(name, i) 
        X = SwinTransformerBlock(dim=embed_dim, num_patch=num_patch, num_heads=num_heads, 
                                 window_size=window_size, shift_size=shift_size_temp, num_mlp=num_mlp, 
                                 qkv_bias=qkv_bias, qk_scale=qk_scale, mlp_drop=mlp_drop_rate, 
                                 attn_drop=attn_drop_rate, proj_drop=proj_drop_rate, 
                                 drop_path_prob=drop_path_rate, name=block_name)(X)
    return X


def swin_unet_2d_base(input_tensor, filter_num_begin, depth, stack_num_down, stack_num_up, 
                      patch_size, num_heads, window_size, num_mlp, shift_window=True, name='swin_unet'):
    
    input_size = input_tensor.shape.as_list()[1:]
    num_patch_x = input_size[0] // patch_size[0]
    num_patch_y = input_size[1] // patch_size[1]
    
    embed_dim = filter_num_begin
    depth_ = depth
    
    X_skip = []

    X = input_tensor
    
    # Patch extraction
    X = PatchExtract(patch_size)(X)

    # Embed patches to tokens
    X = PatchEmbedding(num_patch_x * num_patch_y, embed_dim)(X)
    
    # The first Swin Transformer stack
    X = swin_transformer_stack(X, stack_num=stack_num_down, 
                               embed_dim=embed_dim, num_patch=(num_patch_x, num_patch_y), 
                               num_heads=num_heads[0], window_size=window_size[0], num_mlp=num_mlp, 
                               shift_window=shift_window, name='{}_swinblock_base'.format(name))
    X_skip.append(X)
    
    # Downsampling blocks
    for i in range(depth_ - 1):
        X = CustomPatchMerging(embed_dim=embed_dim, name='{}_down_{}'.format(name, i))(X)
        
        embed_dim *= 2
        num_patch_x //= 2
        num_patch_y //= 2
        
        X = swin_transformer_stack(X, stack_num=stack_num_down, 
                                   embed_dim=embed_dim, num_patch=(num_patch_x, num_patch_y), 
                                   num_heads=num_heads[i + 1], window_size=window_size[i + 1], num_mlp=num_mlp, 
                                   shift_window=shift_window, name='{}_swin_down_{}'.format(name, i + 1))
        
        X_skip.append(X)
    
    X_skip = X_skip[::-1]
    num_heads = num_heads[::-1]
    window_size = window_size[::-1]
    
    X = X_skip[0]
    X_decode = X_skip[1:]
    depth_decode = len(X_decode)
    
    for i in range(depth_decode):
        X = CustomPatchExpanding(embed_dim=embed_dim, name='{}_up_{}'.format(name, i))(X)

        embed_dim //= 2
        num_patch_x *= 2
        num_patch_y *= 2
        
        X = layers.concatenate([X, X_decode[i]], axis=-1, name='{}_concat_{}'.format(name, i))
        X = layers.Dense(embed_dim, use_bias=False, name='{}_concat_linear_proj_{}'.format(name, i))(X)
        
        X = swin_transformer_stack(X, stack_num=stack_num_up, 
                                   embed_dim=embed_dim, num_patch=(num_patch_x, num_patch_y), 
                                   num_heads=num_heads[i], window_size=window_size[i], num_mlp=num_mlp, 
                                   shift_window=shift_window, name='{}_swinblock_up_{}'.format(name, i))
    

    
    return X


# Hauptmodell für SwinUNET
def swin_unet_2d(input_size, filter_num_begin, n_labels, depth, stack_num_down, stack_num_up, 
                  patch_size, num_heads, window_size, num_mlp, output_activation='sigmoid', shift_window=True, name='swin_unet'):
    IN = layers.Input(input_size)
    
    # Basis-Swin-Unet-Modell
    X = swin_unet_2d_base(IN, filter_num_begin=filter_num_begin, depth=depth, stack_num_down=stack_num_down, 
                          stack_num_up=stack_num_up, patch_size=patch_size, num_heads=num_heads, 
                          window_size=window_size, num_mlp=num_mlp, shift_window=shift_window, name=name)
    
    # Sicherstellen, dass die Form korrekt ist: (None, 64, 64, 512)
    X = layers.Conv2D(n_labels, kernel_size=1, activation=None, name='reduce_channels_to_n_labels')(X)  # → (None, 64, 64, 2)
    
    # Upsampling von 64x64 auf 512x512
    X = layers.UpSampling2D(size=(8, 8), interpolation='bilinear', name='upsample_to_512x512' ,data_format="channels_last" )(X)  # → (None, 512, 512, 2)
    
    # Finale Schicht
    OUT = layers.Conv2D(n_labels, kernel_size=1, activation=output_activation, name='{}_outlayer'.format(name))(X)
   # OUT = layers.Permute((2, 3, 1))(OUT) 

    # Erstelle das Modell im funktionalen API-Stil
    model = models.Model(inputs=[IN], outputs=[OUT], name='model_{}_depth{}'.format(name, depth))
    
    return model

model = swin_unet_2d(
    input_size=(512, 512, 3), 
    filter_num_begin=512, 
    n_labels=1, 
    depth=4, 
    stack_num_down=4, 
    stack_num_up=4, 
    patch_size=(8, 8), 
    num_heads=[4, 8, 16, 16],  
    window_size=[64, 64, 16, 16],  
    num_mlp=2, 
    output_activation='sigmoid', 
    shift_window=True, 
    name='swin_unet'
)
model.build(input_shape=(None, 512, 512, 3)) 
model.summary()
