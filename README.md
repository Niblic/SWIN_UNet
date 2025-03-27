Swin Unet as discussed in:
Cao, H. et al. (2023). Swin-Unet: Unet-Like Pure Transformer for Medical Image Segmentation. In: Karlinsky, L., Michaeli, T., Nishino, K. (eds) Computer Vision – ECCV 2022 Workshops. ECCV 2022. Lecture Notes in Computer Science, vol 13803. Springer, Cham. https://doi.org/10.1007/978-3-031-25066-8_9

was portet to run with Tensorflow 2.16
The network generated is for use with 512x512 images and can be changed.

The return is either mulitclass or binary.
Currently its set up for binary


The Swin Unet has the following layout:
Model: "model_swin_unet_depth4"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 input_2 (InputLayer)        [(None, 512, 512, 3)]        0         []                            
                                                                                                  
 patch_extract_1 (PatchExtr  (None, 64, 64, 192)          0         ['input_2[0][0]']             
 act)                                                                                             
                                                                                                  
 patch_embedding_1 (PatchEm  (None, 64, 64, 512)          98816     ['patch_extract_1[0][0]']     
 bedding)                                                                                         
                                                                                                  
 swin_unet_swinblock_base_s  (None, 64, 64, 512)          0         ['patch_embedding_1[0][0]']   
 win_block_0 (SwinTransform                                                                       
 erBlock)                                                                                         
                                                                                                  
 swin_unet_swinblock_base_s  (None, 64, 64, 512)          0         ['swin_unet_swinblock_base_swi
 win_block_1 (SwinTransform                                         n_block_0[0][0]']             
 erBlock)                                                                                         
                                                                                                  
 swin_unet_swinblock_base_s  (None, 64, 64, 512)          0         ['swin_unet_swinblock_base_swi
 win_block_2 (SwinTransform                                         n_block_1[0][0]']             
 erBlock)                                                                                         
                                                                                                  
 swin_unet_swinblock_base_s  (None, 64, 64, 512)          0         ['swin_unet_swinblock_base_swi
 win_block_3 (SwinTransform                                         n_block_2[0][0]']             
 erBlock)                                                                                         
                                                                                                  
 swin_unet_down_0 (CustomPa  (None, 32, 32, 1024)         524288    ['swin_unet_swinblock_base_swi
 tchMerging)                                                        n_block_3[0][0]']             
                                                                                                  
 swin_unet_swin_down_1_swin  (None, 32, 32, 1024)         0         ['swin_unet_down_0[0][0]']    
 _block_0 (SwinTransformerB                                                                       
 lock)                                                                                            
                                                                                                  
 swin_unet_swin_down_1_swin  (None, 32, 32, 1024)         0         ['swin_unet_swin_down_1_swin_b
 _block_1 (SwinTransformerB                                         lock_0[0][0]']                
 lock)                                                                                            
                                                                                                  
 swin_unet_swin_down_1_swin  (None, 32, 32, 1024)         0         ['swin_unet_swin_down_1_swin_b
 _block_2 (SwinTransformerB                                         lock_1[0][0]']                
 lock)                                                                                            
                                                                                                  
 swin_unet_swin_down_1_swin  (None, 32, 32, 1024)         0         ['swin_unet_swin_down_1_swin_b
 _block_3 (SwinTransformerB                                         lock_2[0][0]']                
 lock)                                                                                            
                                                                                                  
 swin_unet_down_1 (CustomPa  (None, 16, 16, 2048)         2097152   ['swin_unet_swin_down_1_swin_b
 tchMerging)                                                        lock_3[0][0]']                
                                                                                                  
 swin_unet_swin_down_2_swin  (None, 16, 16, 2048)         0         ['swin_unet_down_1[0][0]']    
 _block_0 (SwinTransformerB                                                                       
 lock)                                                                                            
                                                                                                  
 swin_unet_swin_down_2_swin  (None, 16, 16, 2048)         0         ['swin_unet_swin_down_2_swin_b
 _block_1 (SwinTransformerB                                         lock_0[0][0]']                
 lock)                                                                                            
                                                                                                  
 swin_unet_swin_down_2_swin  (None, 16, 16, 2048)         0         ['swin_unet_swin_down_2_swin_b
 _block_2 (SwinTransformerB                                         lock_1[0][0]']                
 lock)                                                                                            
                                                                                                  
 swin_unet_swin_down_2_swin  (None, 16, 16, 2048)         0         ['swin_unet_swin_down_2_swin_b
 _block_3 (SwinTransformerB                                         lock_2[0][0]']                
 lock)                                                                                            
                                                                                                  
 swin_unet_down_2 (CustomPa  (None, 8, 8, 4096)           8388608   ['swin_unet_swin_down_2_swin_b
 tchMerging)                                                        lock_3[0][0]']                
                                                                                                  
 swin_unet_swin_down_3_swin  (None, 8, 8, 4096)           0         ['swin_unet_down_2[0][0]']    
 _block_0 (SwinTransformerB                                                                       
 lock)                                                                                            
                                                                                                  
 swin_unet_swin_down_3_swin  (None, 8, 8, 4096)           0         ['swin_unet_swin_down_3_swin_b
 _block_1 (SwinTransformerB                                         lock_0[0][0]']                
 lock)                                                                                            
                                                                                                  
 swin_unet_swin_down_3_swin  (None, 8, 8, 4096)           0         ['swin_unet_swin_down_3_swin_b
 _block_2 (SwinTransformerB                                         lock_1[0][0]']                
 lock)                                                                                            
                                                                                                  
 swin_unet_swin_down_3_swin  (None, 8, 8, 4096)           0         ['swin_unet_swin_down_3_swin_b
 _block_3 (SwinTransformerB                                         lock_2[0][0]']                
 lock)                                                                                            
                                                                                                  
 swin_unet_up_0 (CustomPatc  (None, 16, 16, 4096)         1677721   ['swin_unet_swin_down_3_swin_b
 hExpanding)                                              6         lock_3[0][0]']                
                                                                                                  
 swin_unet_concat_0 (Concat  (None, 16, 16, 6144)         0         ['swin_unet_up_0[0][0]',      
 enate)                                                              'swin_unet_swin_down_2_swin_b
                                                                    lock_3[0][0]']                
                                                                                                  
 swin_unet_concat_linear_pr  (None, 16, 16, 2048)         1258291   ['swin_unet_concat_0[0][0]']  
 oj_0 (Dense)                                             2                                       
                                                                                                  
 swin_unet_swinblock_up_0_s  (None, 16, 16, 2048)         0         ['swin_unet_concat_linear_proj
 win_block_0 (SwinTransform                                         _0[0][0]']                    
 erBlock)                                                                                         
                                                                                                  
 swin_unet_swinblock_up_0_s  (None, 16, 16, 2048)         0         ['swin_unet_swinblock_up_0_swi
 win_block_1 (SwinTransform                                         n_block_0[0][0]']             
 erBlock)                                                                                         
                                                                                                  
 swin_unet_swinblock_up_0_s  (None, 16, 16, 2048)         0         ['swin_unet_swinblock_up_0_swi
 win_block_2 (SwinTransform                                         n_block_1[0][0]']             
 erBlock)                                                                                         
                                                                                                  
 swin_unet_swinblock_up_0_s  (None, 16, 16, 2048)         0         ['swin_unet_swinblock_up_0_swi
 win_block_3 (SwinTransform                                         n_block_2[0][0]']             
 erBlock)                                                                                         
                                                                                                  
 swin_unet_up_1 (CustomPatc  (None, 32, 32, 2048)         4194304   ['swin_unet_swinblock_up_0_swi
 hExpanding)                                                        n_block_3[0][0]']             
                                                                                                  
 swin_unet_concat_1 (Concat  (None, 32, 32, 3072)         0         ['swin_unet_up_1[0][0]',      
 enate)                                                              'swin_unet_swin_down_1_swin_b
                                                                    lock_3[0][0]']                
                                                                                                  
 swin_unet_concat_linear_pr  (None, 32, 32, 1024)         3145728   ['swin_unet_concat_1[0][0]']  
 oj_1 (Dense)                                                                                     
                                                                                                  
 swin_unet_swinblock_up_1_s  (None, 32, 32, 1024)         0         ['swin_unet_concat_linear_proj
 win_block_0 (SwinTransform                                         _1[0][0]']                    
 erBlock)                                                                                         
                                                                                                  
 swin_unet_swinblock_up_1_s  (None, 32, 32, 1024)         0         ['swin_unet_swinblock_up_1_swi
 win_block_1 (SwinTransform                                         n_block_0[0][0]']             
 erBlock)                                                                                         
                                                                                                  
 swin_unet_swinblock_up_1_s  (None, 32, 32, 1024)         0         ['swin_unet_swinblock_up_1_swi
 win_block_2 (SwinTransform                                         n_block_1[0][0]']             
 erBlock)                                                                                         
                                                                                                  
 swin_unet_swinblock_up_1_s  (None, 32, 32, 1024)         0         ['swin_unet_swinblock_up_1_swi
 win_block_3 (SwinTransform                                         n_block_2[0][0]']             
 erBlock)                                                                                         
                                                                                                  
 swin_unet_up_2 (CustomPatc  (None, 64, 64, 1024)         1048576   ['swin_unet_swinblock_up_1_swi
 hExpanding)                                                        n_block_3[0][0]']             
                                                                                                  
 swin_unet_concat_2 (Concat  (None, 64, 64, 1536)         0         ['swin_unet_up_2[0][0]',      
 enate)                                                              'swin_unet_swinblock_base_swi
                                                                    n_block_3[0][0]']             
                                                                                                  
 swin_unet_concat_linear_pr  (None, 64, 64, 512)          786432    ['swin_unet_concat_2[0][0]']  
 oj_2 (Dense)                                                                                     
                                                                                                  
 swin_unet_swinblock_up_2_s  (None, 64, 64, 512)          0         ['swin_unet_concat_linear_proj
 win_block_0 (SwinTransform                                         _2[0][0]']                    
 erBlock)                                                                                         
                                                                                                  
 swin_unet_swinblock_up_2_s  (None, 64, 64, 512)          0         ['swin_unet_swinblock_up_2_swi
 win_block_1 (SwinTransform                                         n_block_0[0][0]']             
 erBlock)                                                                                         
                                                                                                  
 swin_unet_swinblock_up_2_s  (None, 64, 64, 512)          0         ['swin_unet_swinblock_up_2_swi
 win_block_2 (SwinTransform                                         n_block_1[0][0]']             
 erBlock)                                                                                         
                                                                                                  
 swin_unet_swinblock_up_2_s  (None, 64, 64, 512)          0         ['swin_unet_swinblock_up_2_swi
 win_block_3 (SwinTransform                                         n_block_2[0][0]']             
 erBlock)                                                                                         
                                                                                                  
 reduce_channels_to_n_label  (None, 64, 64, 1)            513       ['swin_unet_swinblock_up_2_swi
 s (Conv2D)                                                         n_block_3[0][0]']             
                                                                                                  
 upsample_to_512x512 (UpSam  (None, 512, 512, 1)          0         ['reduce_channels_to_n_labels[
 pling2D)                                                           0][0]']                       
                                                                                                  
 swin_unet_outlayer (Conv2D  (None, 512, 512, 1)          2         ['upsample_to_512x512[0][0]'] 
 )                                                                                                
                                                                                                  
==================================================================================================
Total params: 49644547 (189.38 MB)
Trainable params: 49644547 (189.38 MB)
Non-trainable params: 0 (0.00 Byte)


