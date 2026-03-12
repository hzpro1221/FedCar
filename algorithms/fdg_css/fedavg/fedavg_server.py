

class FedAvg_Server:
    def __init__(
        self, 
        backbone_model, 
        source_domains,
        num_rounds, 
        num_epochs, 
        batch_size,

        init_lr,
        min_lr,
        power,
        weight_decay,
        seed
    ):
    """
    1. backbone_model: The instance of backbone model's class. In this work, it's fixed as SegmentFormer-B0.
    2. source_domains: A list of source domains used to train (for simplicity, each client will correspond with a domain).
    3. num_rounds: Number of communication rounds.
    4. num_epochs: Number of epoch (used to train in server side).
    5. batch_size: Number of batch size (also used in server side).
    
    6. init_lr & min_lr & power: used to schedule learning rate.
    7. weight_decay: Used in AdamW optimizer (in this work, by default the optimizer will be AdamW optimizer)
    8. seed: the seed used in this run.
    """
        self.backbone_model = backbone_model
        self.source_domains = source_domains
        self.num_rounds = num_rounds
        self.num_epochs = num_epochs
        
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.power = power
        self.weight_decay = weight_decay
        self.seed = seed

        # :vv trivial.. but this is for identifing device (in case you don't know)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # For each domain -> we init a client, and asign a domain to him
        self.clients = []
        for domain in self.source_domains:
            self.clients.append(
                
            )
    
    def set_seed(self): 
        # for python and numpy
        random.seed(self.seed)
        np.random.seed(self.seed)

        # pytorch cpu & gpu (this is only for single GPU)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
    def train():
        # code for train