Caltech_class = (
    "ak47", "american-flag", "backpack", "baseball-bat", "baseball-glove", "basketball-hoop", "bat", "bathtub", "bear",
    "beer-mug", "billiards", "binoculars", "birdbath", "blimp", "bonsai-101", "boom-box", "bowling-ball",
    "bowling-pin", "boxing-glove", "brain-101", "breadmaker", "buddha-101", "bulldozer", "butterfly", "cactus", "cake",
    "calculator", "camel", "cannon", "canoe", "car-tire", "cartman", "cd", "centipede", "cereal-box", "chandelier-101",
    "chess-board", "chimp", "chopsticks", "cockroach", "coffee-mug", "coffin", "coin", "comet", "computer-keyboard",
    "computer-monitor", "computer-mouse", "conch", "cormorant", "covered-wagon", "cowboy-hat", "crab-101",
    "desk-globe", "diamond-ring", "dice", "dog", "dolphin-101", "doorknob", "drinking-straw", "duck", "dumb-bell",
    "eiffel-tower", "electric-guitar-101", "elephant-101", "elk", "ewer-101", "eyeglasses", "fern", "fighter-jet",
    "fire-extinguisher", "fire-hydrant", "fire-truck", "fireworks", "flashlight", "floppy-disk", "football-helmet",
    "french-horn", "fried-egg", "frisbee", "frog", "frying-pan", "galaxy", "gas-pump", "giraffe", "goat",
    "golden-gate-bridge", "goldfish", "golf-ball", "goose", "gorilla", "grand-piano-101", "grapes", "grasshopper",
    "guitar-pick", "hamburger", "hammock", "harmonica", "harp", "harpsichord", "hawksbill-101", "head-phones",
    "helicopter-101", "hibiscus", "homer-simpson", "horse", "horseshoe-crab", "hot-air-balloon", "hot-dog", "hot-tub",
    "hourglass", "house-fly", "human-skeleton", "hummingbird", "ibis-101", "ice-cream-cone", "iguana", "ipod", "iris",
    "jesus-christ", "joy-stick", "kangaroo-101", "kayak", "ketch-101", "killer-whale", "knife", "ladder", "laptop-101",
    "lathe", "leopards-101", "license-plate", "lightbulb", "light-house", "lightning", "llama-101", "mailbox",
    "mandolin", "mars", "mattress", "megaphone", "menorah-101", "microscope", "microwave", "minaret", "minotaur",
    "motorbikes-101", "mountain-bike", "mushroom", "mussels", "necktie", "octopus", "ostrich", "owl", "palm-pilot",
    "palm-tree", "paperclip", "paper-shredder", "pci-card", "penguin", "people", "pez-dispenser", "photocopier",
    "picnic-table", "playing-card", "porcupine", "pram", "praying-mantis", "pyramid", "raccoon", "radio-telescope",
    "rainbow", "refrigerator", "revolver-101", "rifle", "rotary-phone", "roulette-wheel", "saddle", "saturn",
    "school-bus", "scorpion-101", "screwdriver", "segway", "self-propelled-lawn-mower", "sextant", "sheet-music",
    "skateboard", "skunk", "skyscraper", "smokestack", "snail", "snake", "sneaker", "snowmobile", "soccer-ball",
    "socks", "soda-can", "spaghetti", "speed-boat", "spider", "spoon", "stained-glass", "starfish-101",
    "steering-wheel", "stirrups", "sunflower-101", "superman", "sushi", "swan", "swiss-army-knife", "sword", "syringe",
    "tambourine", "teapot", "teddy-bear", "teepee", "telephone-box", "tennis-ball", "tennis-court", "tennis-racket",
    "theodolite", "toaster", "tomato", "tombstone", "top-hat", "touring-bike", "tower-pisa", "traffic-light",
    "treadmill", "triceratops", "tricycle", "trilobite-101", "tripod", "t-shirt", "tuning-fork", "tweezer",
    "umbrella-101", "unicorn", "vcr", "video-projector", "washing-machine", "watch-101", "waterfall", "watermelon",
    "welding-mask", "wheelbarrow", "windmill", "wine-bottle", "xylophone", "yarmulke", "yo-yo", "zebra",
    "airplanes-101", "car-side-101", "faces-easy-101", "greyhound", "tennis-shoes", "toad", "clutter"
)


class Config(object):
    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        ret = Config(vars(self))

        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict):
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)

    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v)


dataset_base = Config({
    'name': 'Caltech256',
    'class_names': Caltech_class,
    "train": "/data/yang/openData/Caltech256/train.lst",
    "val": "/data/yang/openData/Caltech256/val.lst",
    "test": "/data/yang/openData/Caltech256/test.lst",
    "root": "/data/yang/openData/Caltech256"
})

train_config = Config({
    'resize': (224, 224),
    'padding': (127, 127, 127),
    "MEAN": (0.408, 0.458, 0.485),
    "STD": (0.255, 0.224, 0.229),
    "augment_horizontal_flip": True,
    "augment_vertical_flip": True,
    "augment_rotation": True,
    "epoch": 30,
    "base_lr": 0.01,
    "batch_size": 64,
    "start_epoch": 0,
    "saveName": "efficientnet-b2",
    "resume": "",
    "weightDir": "output/weight_saved/",
    "logDir": "output/log"


})

val_config = Config({
    'resize': (224, 224),
    'padding': (127, 127, 127),
    "MEAN": (0.408, 0.458, 0.485),  # PIL的值是0-1
    "STD": (0.255, 0.224, 0.229),
    "augment_horizontal_flip": False,
    "augment_vertical_flip": False,
    "augment_rotation": False,
    "batch_size": 16,
})
