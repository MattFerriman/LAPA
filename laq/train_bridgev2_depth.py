from laq_model import LAQTrainerWithDepth
from laq_model import LatentActionQuantization



laq = LatentActionQuantization(
    dim = 1024,
    quant_dim=32,
    codebook_size = 8,
    image_size = 256,
    patch_size = 32,
    spatial_depth = 8, #8
    temporal_depth = 8, #8
    dim_head = 64,
    heads = 16,
    code_seq_len=4,
    channels = 4,
).cuda()


trainer = LAQTrainerWithDepth(
    laq,
    folder = '',
    offsets = 5,
    batch_size = 64,
    grad_accum_every = 1,
    train_on_images = False, 
    use_ema = False,          
    num_train_steps = 30005,
    results_folder='results_tk1257_mv_depth',
    lr=1e-4,
    save_model_every=5000,
    save_results_every=5000,
)

trainer.train()        
