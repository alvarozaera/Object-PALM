coco_classes = {0: u'__background__',
 1: u'person',
 2: u'bicycle',
 3: u'car',
 4: u'motorcycle',
 5: u'airplane',
 6: u'bus',
 7: u'train',
 8: u'truck',
 9: u'boat',
 10: u'traffic light',
 11: u'fire hydrant',
 12: u'stop sign',
 13: u'parking meter',
 14: u'bench',
 15: u'bird',
 16: u'cat',
 17: u'dog',
 18: u'horse',
 19: u'sheep',
 20: u'cow',
 21: u'elephant',
 22: u'bear',
 23: u'zebra',
 24: u'giraffe',
 25: u'backpack',
 26: u'umbrella',
 27: u'handbag',
 28: u'tie',
 29: u'suitcase',
 30: u'frisbee',
 31: u'skis',
 32: u'snowboard',
 33: u'sports ball',
 34: u'kite',
 35: u'baseball bat',
 36: u'baseball glove',
 37: u'skateboard',
 38: u'surfboard',
 39: u'tennis racket',
 40: u'bottle',
 41: u'wine glass',
 42: u'cup',
 43: u'fork',
 44: u'knife',
 45: u'spoon',
 46: u'bowl',
 47: u'banana',
 48: u'apple',
 49: u'sandwich',
 50: u'orange',
 51: u'broccoli',
 52: u'carrot',
 53: u'hot dog',
 54: u'pizza',
 55: u'donut',
 56: u'cake',
 57: u'chair',
 58: u'couch',
 59: u'potted plant',
 60: u'bed',
 61: u'dining table',
 62: u'toilet',
 63: u'tv',
 64: u'laptop',
 65: u'mouse',
 66: u'remote',
 67: u'keyboard',
 68: u'cell phone',
 69: u'microwave',
 70: u'oven',
 71: u'toaster',
 72: u'sink',
 73: u'refrigerator',
 74: u'book',
 75: u'clock',
 76: u'vase',
 77: u'scissors',
 78: u'teddy bear',
 79: u'hair drier',
 80: u'toothbrush',
}

coco_mapping = {
    "airplane" : "plane",
    "bus" : "car",
    "traffic light" : "light",
    "backpack" : "bag",
    "handbag" : "bag",
    "sportsball" : "ball",
    "baseball bat" : "bat",
    "baseball glove" : "glove",
    "tennis racket" : "racket",
    "wine glass": "glass",
    "hot dog" : "food",
    "donut" : "food",
    "couch" : "sofa",
    "potted plant" : "plant",
    "dinning table" : "table",
    "cell phone" : "phone",
}

obj365_raw = """
  0: Person
  1: Sneakers
  2: Chair
  3: Other Shoes
  4: Hat
  5: Car
  6: Lamp
  7: Glasses
  8: Bottle
  9: Desk
  10: Cup
  11: Street Lights
  12: Cabinet/shelf
  13: Handbag/Satchel
  14: Bracelet
  15: Plate
  16: Picture/Frame
  17: Helmet
  18: Book
  19: Gloves
  20: Storage box
  21: Boat
  22: Leather Shoes
  23: Flower
  24: Bench
  25: Potted Plant
  26: Bowl/Basin
  27: Flag
  28: Pillow
  29: Boots
  30: Vase
  31: Microphone
  32: Necklace
  33: Ring
  34: SUV
  35: Wine Glass
  36: Belt
  37: Monitor/TV
  38: Backpack
  39: Umbrella
  40: Traffic Light
  41: Speaker
  42: Watch
  43: Tie
  44: Trash bin Can
  45: Slippers
  46: Bicycle
  47: Stool
  48: Barrel/bucket
  49: Van
  50: Couch
  51: Sandals
  52: Basket
  53: Drum
  54: Pen/Pencil
  55: Bus
  56: Wild Bird
  57: High Heels
  58: Motorcycle
  59: Guitar
  60: Carpet
  61: Cell Phone
  62: Bread
  63: Camera
  64: Canned
  65: Truck
  66: Traffic cone
  67: Cymbal
  68: Lifesaver
  69: Towel
  70: Stuffed Toy
  71: Candle
  72: Sailboat
  73: Laptop
  74: Awning
  75: Bed
  76: Faucet
  77: Tent
  78: Horse
  79: Mirror
  80: Power outlet
  81: Sink
  82: Apple
  83: Air Conditioner
  84: Knife
  85: Hockey Stick
  86: Paddle
  87: Pickup Truck
  88: Fork
  89: Traffic Sign
  90: Balloon
  91: Tripod
  92: Dog
  93: Spoon
  94: Clock
  95: Pot
  96: Cow
  97: Cake
  98: Dinning Table
  99: Sheep
  100: Hanger
  101: Blackboard/Whiteboard
  102: Napkin
  103: Other Fish
  104: Orange/Tangerine
  105: Toiletry
  106: Keyboard
  107: Tomato
  108: Lantern
  109: Machinery Vehicle
  110: Fan
  111: Green Vegetables
  112: Banana
  113: Baseball Glove
  114: Airplane
  115: Mouse
  116: Train
  117: Pumpkin
  118: Soccer
  119: Skiboard
  120: Luggage
  121: Nightstand
  122: Tea pot
  123: Telephone
  124: Trolley
  125: Head Phone
  126: Sports Car
  127: Stop Sign
  128: Dessert
  129: Scooter
  130: Stroller
  131: Crane
  132: Remote
  133: Refrigerator
  134: Oven
  135: Lemon
  136: Duck
  137: Baseball Bat
  138: Surveillance Camera
  139: Cat
  140: Jug
  141: Broccoli
  142: Piano
  143: Pizza
  144: Elephant
  145: Skateboard
  146: Surfboard
  147: Gun
  148: Skating and Skiing shoes
  149: Gas stove
  150: Donut
  151: Bow Tie
  152: Carrot
  153: Toilet
  154: Kite
  155: Strawberry
  156: Other Balls
  157: Shovel
  158: Pepper
  159: Computer Box
  160: Toilet Paper
  161: Cleaning Products
  162: Chopsticks
  163: Microwave
  164: Pigeon
  165: Baseball
  166: Cutting/chopping Board
  167: Coffee Table
  168: Side Table
  169: Scissors
  170: Marker
  171: Pie
  172: Ladder
  173: Snowboard
  174: Cookies
  175: Radiator
  176: Fire Hydrant
  177: Basketball
  178: Zebra
  179: Grape
  180: Giraffe
  181: Potato
  182: Sausage
  183: Tricycle
  184: Violin
  185: Egg
  186: Fire Extinguisher
  187: Candy
  188: Fire Truck
  189: Billiards
  190: Converter
  191: Bathtub
  192: Wheelchair
  193: Golf Club
  194: Briefcase
  195: Cucumber
  196: Cigar/Cigarette
  197: Paint Brush
  198: Pear
  199: Heavy Truck
  200: Hamburger
  201: Extractor
  202: Extension Cord
  203: Tong
  204: Tennis Racket
  205: Folder
  206: American Football
  207: earphone
  208: Mask
  209: Kettle
  210: Tennis
  211: Ship
  212: Swing
  213: Coffee Machine
  214: Slide
  215: Carriage
  216: Onion
  217: Green beans
  218: Projector
  219: Frisbee
  220: Washing Machine/Drying Machine
  221: Chicken
  222: Printer
  223: Watermelon
  224: Saxophone
  225: Tissue
  226: Toothbrush
  227: Ice cream
  228: Hot-air balloon
  229: Cello
  230: French Fries
  231: Scale
  232: Trophy
  233: Cabbage
  234: Hot dog
  235: Blender
  236: Peach
  237: Rice
  238: Wallet/Purse
  239: Volleyball
  240: Deer
  241: Goose
  242: Tape
  243: Tablet
  244: Cosmetics
  245: Trumpet
  246: Pineapple
  247: Golf Ball
  248: Ambulance
  249: Parking meter
  250: Mango
  251: Key
  252: Hurdle
  253: Fishing Rod
  254: Medal
  255: Flute
  256: Brush
  257: Penguin
  258: Megaphone
  259: Corn
  260: Lettuce
  261: Garlic
  262: Swan
  263: Helicopter
  264: Green Onion
  265: Sandwich
  266: Nuts
  267: Speed Limit Sign
  268: Induction Cooker
  269: Broom
  270: Trombone
  271: Plum
  272: Rickshaw
  273: Goldfish
  274: Kiwi fruit
  275: Router/modem
  276: Poker Card
  277: Toaster
  278: Shrimp
  279: Sushi
  280: Cheese
  281: Notepaper
  282: Cherry
  283: Pliers
  284: CD
  285: Pasta
  286: Hammer
  287: Cue
  288: Avocado
  289: Hamimelon
  290: Flask
  291: Mushroom
  292: Screwdriver
  293: Soap
  294: Recorder
  295: Bear
  296: Eggplant
  297: Board Eraser
  298: Coconut
  299: Tape Measure/Ruler
  300: Pig
  301: Showerhead
  302: Globe
  303: Chips
  304: Steak
  305: Crosswalk Sign
  306: Stapler
  307: Camel
  308: Formula 1
  309: Pomegranate
  310: Dishwasher
  311: Crab
  312: Hoverboard
  313: Meat ball
  314: Rice Cooker
  315: Tuba
  316: Calculator
  317: Papaya
  318: Antelope
  319: Parrot
  320: Seal
  321: Butterfly
  322: Dumbbell
  323: Donkey
  324: Lion
  325: Urinal
  326: Dolphin
  327: Electric Drill
  328: Hair Dryer
  329: Egg tart
  330: Jellyfish
  331: Treadmill
  332: Lighter
  333: Grapefruit
  334: Game board
  335: Mop
  336: Radish
  337: Baozi
  338: Target
  339: French
  340: Spring Rolls
  341: Monkey
  342: Rabbit
  343: Pencil Case
  344: Yak
  345: Red Cabbage
  346: Binoculars
  347: Asparagus
  348: Barbell
  349: Scallop
  350: Noddles
  351: Comb
  352: Dumpling
  353: Oyster
  354: Table Tennis paddle
  355: Cosmetics Brush/Eyeliner Pencil
  356: Chainsaw
  357: Eraser
  358: Lobster
  359: Durian
  360: Okra
  361: Lipstick
  362: Cosmetics Mirror
  363: Curling
  364: Table Tennis
"""
obj365_classes = [l.split(": ")[-1].lower() for l in obj365_raw.split("\n")[1:-1]]

obj365_mapping = {
    "sneakers" : "shoe",
    "other shoes" : "shoe",
    "desk" : "table",
    "cabinet/shelf" : "cabinet",
    "handbag/satchel" : "bag",
    "picture/frame" : "picture",
    "gloves" : "glove",
    "storage box" : "container",
    "leather shoes" : "shoe",
    "potted plant" : "plant",
    "bowl/basin" : "bowl",
    "boots" : "boot",
    "suv" : "car",
    "wine glass" : "glass",
    "monitor/tv" : "tv",
    "backpack" : "bag",
    "trash bin can" : "bin",
    "slippers" : "shoe",
    "barrel/bucket" : "bucket",
    "van" : "car",
    "couch" : "sofa",
    "sandals" : "shoe",
    "pen/pencil" : "pen",
    "bus" : "car",
    "wild bird" : "bird",
    "high heels" : "shoe",
    "cell phone" : "phone",
    "stuffed toy" : "toy",
    "hockey stick" : "stick",
    "pickup truck" : "car",
    "dinning table" : "table",
    "blackboard/whiteboard" : "cardboard",
    "other fish" : "fish",
    "orange/tangerine" : "orange",
    "toiletry" : "bag",
    "green vegetables" : "vegetable",
    "baseball glove" : "glove",
    "airplane" : "plane",
    "soccer" : "ball",
    "table tennis" : "ball",
    "cosmetics mirror" : "mirror",
    "lipstick" : "stick",
    "dumpling" : "food",
    "noddles" : "food",
    "red cabbage" : "vegetable",
    "poker card" : "card",
    "nuts" : "nut",
    "board" : "wood",
    "telephone" : "phone",
    "paint brush" : "paintbrush",
    "extension cord" : "cord",
}

obj365_mapping.update(coco_mapping)

unidet_classes = ['bottle',
 'cup',
 'dining table',
 'cake',
 'wild bird',
 'knife',
 'remote',
 'microwave',
 'frisbee',
 'toothbrush',
 'Footwear',
 'Picture frame',
 'Table',
 'Street light',
 'Book',
 'Helmet',
 'Pillow',
 'Box',
 'Bowl',
 'Watercraft',
 'Flag',
 'Stool',
 'Doll',
 'Pen',
 'Microphone',
 'Tap',
 'Bread',
 'Sandal',
 'Fish',
 'Camera',
 'Candle',
 'Paddle',
 'Drum',
 'Guitar',
 'Kettle',
 'Ceiling fan',
 'Whiteboard',
 'Balloon',
 'Corded phone',
 'Orange',
 'Football',
 'Toilet paper',
 'Tomato',
 'Tent',
 'Lantern',
 'Kite',
 'Gas stove',
 'Spatula',
 'Rifle',
 'Lemon',
 'Squash',
 'Musical keyboard',
 'Washing machine',
 'Cookie',
 'Cutting board',
 'Roller skates',
 'Cricket ball',
 'Strawberry',
 'Coffeemaker',
 'Suitcase',
 'Grape',
 'Ladder',
 'Pear',
 'Rugby ball',
 'Printer',
 'Duck',
 'Tennis ball',
 'Chopsticks',
 'Hamburger',
 'Cucumber',
 'Mixer',
 'Deer',
 'Egg',
 'Barge',
 'Turkey',
 'Ice cream',
 'Adhesive tape',
 'Wheelchair',
 'Cabbage',
 'Golf ball',
 'Peach',
 'Cello',
 'Helicopter',
 'Penguin',
 'Swan',
 'French fries',
 'Saxophone',
 'Trombone',
 'Raccoon',
 'Tablet computer',
 'Volleyball',
 'Dumbbell',
 'Camel',
 'Goldfish',
 'Antelope',
 'Shrimp',
 'Cart',
 'Coconut',
 'Jellyfish',
 'Treadmill',
 'Butterfly',
 'Pig',
 'Shower',
 'Asparagus',
 'Dolphin',
 'Sushi',
 'Burrito',
 'Tortoise',
 'Parrot',
 'Flute',
 'Shark',
 'Binoculars',
 'Alpaca',
 'Pasta',
 'Shellfish',
 'Lion',
 'Polar bear',
 'Sea lion',
 'Table tennis racket',
 'Starfish',
 'Falcon',
 'Monkey',
 'Rabbit',
 'Ambulance',
 'Segway',
 'Truck',
 'Boat',
 'Bear',
 'Handbag',
 'Ball',
 'Sandwich',
 'Bidet',
 'Computer monitor',
 'Vase',
 'Person',
 'Chair',
 'Car',
 'Houseplant',
 'Bench2',
 'Wine glass',
 'Umbrella',
 'Backpack',
 'Loveseat',
 'Tie',
 'Infant bed',
 'Traffic light',
 'Bicycle',
 'Sink',
 'Horse',
 'Apple',
 'Teddy bear',
 'Motorcycle',
 'Laptop',
 'Mobile phone',
 'Cattle',
 'Clock',
 'Fork',
 'Bus',
 'Sheep',
 'Computer keyboard',
 'Dog',
 'Spoon',
 'Mouse2',
 'Banana',
 'Airplane',
 'Briefcase',
 'Ski',
 'Baseball glove',
 'Refrigerator',
 'Train',
 'Doughnut',
 'Grapefruit',
 'Pizza',
 'Elephant',
 'Broccoli',
 'Baseball bat',
 'Skateboard',
 'Surfboard',
 'Cat',
 'Zebra',
 'Giraffe',
 'Stop sign',
 'Carrot',
 'Tennis racket',
 'Scissors',
 'Snowboard',
 'Fire hydrant',
 'Hot dog',
 'Toaster',
 'hat',
 'lamp',
 'cabinet/shelf',
 'glasses',
 'handbag',
 'plate',
 'leather shoes',
 'glove',
 'bracelet',
 'flower',
 'tv',
 'vase',
 'boots',
 'speaker',
 'trash bin/can',
 'belt',
 'carpet',
 'basket',
 'towel/napkin',
 'slippers',
 'barrel/bucket',
 'coffee table',
 'suv',
 'sandals',
 'canned',
 'necklace',
 'mirror',
 'ring',
 'van',
 'watch',
 'traffic sign',
 'truck',
 'power outlet',
 'hanger',
 'nightstand',
 'pot/pan',
 'traffic cone',
 'tripod',
 'hockey',
 'air conditioner',
 'cymbal',
 'pickup truck',
 'trolley',
 'oven',
 'machinery vehicle',
 'shampoo/shower gel',
 'head phone',
 'cleaning products',
 'sailboat',
 'computer box',
 'toiletries',
 'toilet',
 'stroller',
 'surveillance camera',
 'life saver',
 'liquid soap',
 'duck',
 'sports car',
 'radiator',
 'converter',
 'tissue ',
 'vent',
 'candy',
 'folder',
 'bow tie',
 'pigeon',
 'pepper',
 'bathtub',
 'basketball',
 'potato',
 'paint brush',
 'billiards',
 'projector',
 'sausage',
 'fire extinguisher',
 'extension cord',
 'facial mask',
 'electronic stove and gas stove',
 'pie',
 'kettle',
 'golf club',
 'clutch',
 'tong',
 'slide',
 'facial cleanser',
 'mango',
 'violin',
 'marker',
 'onion',
 'plum',
 'bar soap',
 'scale',
 'watermelon',
 'router/modem',
 'pine apple',
 'crane',
 'fire truck',
 'notepaper',
 'tricycle',
 'green beans',
 'brush',
 'carriage',
 'cigar',
 'earphone',
 'hurdle',
 'swing',
 'radio',
 'CD',
 'parking meter',
 'garlic',
 'horn',
 'avocado',
 'sandwich',
 'cue',
 'kiwi fruit',
 'fishing rod',
 'cherry',
 'green vegetables',
 'nuts',
 'corn',
 'key',
 'screwdriver',
 'globe',
 'broom',
 'pliers',
 'hammer',
 'eggplant',
 'trophy',
 'dates',
 'board eraser',
 'rice',
 'tape measure/ruler',
 'hamimelon',
 'stapler',
 'lettuce',
 'meat balls',
 'medal',
 'toothpaste',
 'trombone',
 'pomegranate',
 'mushroom',
 'calculator',
 'egg tart',
 'cheese',
 'pomelo',
 'race car',
 'rice cooker',
 'tuba',
 'crosswalk sign',
 'papaya',
 'chips',
 'urinal',
 'donkey',
 'electric drill',
 'measuring cup',
 'steak',
 'poker card',
 'radish',
 'yak',
 'mop',
 'microscope',
 'barbell',
 'bread/bun',
 'baozi',
 'red cabbage',
 'lighter',
 'mangosteen',
 'comb',
 'eraser',
 'pitaya',
 'scallop',
 'pencil case',
 'saw',
 'okra',
 'durian',
 'game board',
 'french horn',
 'asparagus',
 'pasta',
 'target',
 'hotair balloon',
 'chainsaw',
 'lobster',
 'iron',
 'flashlight',
 'parking meter',
 'kite',
 'bowl',
 'oven',
 'book',
 'hair drier',
 'Rose',
 'Flashlight',
 'Sea turtle',
 'Animal',
 'Glove',
 'Crocodile',
 'House',
 'Guacamole',
 'Vehicle registration plate',
 'Bench1',
 'Ladybug',
 'Human nose',
 'Watermelon',
 'Taco',
 'Cake',
 'Cannon',
 'Tree',
 'Bed',
 'Hamster',
 'Hat',
 'Sombrero',
 'Tiara',
 'Dragonfly',
 'Moths and butterflies',
 'Vegetable',
 'Torch',
 'Building',
 'Power plugs and sockets',
 'Blender',
 'Billiard table',
 'Bronze sculpture',
 'Turtle',
 'Tiger',
 'Mirror',
 'Zucchini',
 'Dress',
 'Reptile',
 'Golf cart',
 'Tart',
 'Fedora',
 'Carnivore',
 'Lighthouse',
 'Food processor',
 'Bookcase',
 'Necklace',
 'Flower',
 'Radish',
 'Marine mammal',
 'Frying pan',
 'Knife',
 'Christmas tree',
 'Eagle',
 'Limousine',
 'Kitchen & dining room table',
 'Tower',
 'Willow',
 'Human head',
 'Dessert',
 'Bee',
 'Wood-burning stove',
 'Flowerpot',
 'Beaker',
 'Oyster',
 'Woodpecker',
 'Harp',
 'Bathtub',
 'Wall clock',
 'Sports uniform',
 'Rhinoceros',
 'Beehive',
 'Cupboard',
 'Chicken',
 'Man',
 'Blue jay',
 'Fireplace',
 'Missile',
 'Squirrel',
 'Coat',
 'Punching bag',
 'Billboard',
 'Door handle',
 'Mechanical fan',
 'Ring binder',
 'Sock',
 'Weapon',
 'Shotgun',
 'Glasses',
 'Seahorse',
 'Belt',
 'Window',
 'Tire',
 'Vehicle',
 'Canoe',
 'Shelf',
 'Human leg',
 'Slow cooker',
 'Croissant',
 'Pancake',
 'Coin',
 'Stretcher',
 'Woman',
 'Stairs',
 'Harpsichord',
 'Human mouth',
 'Juice',
 'Skull',
 'Door',
 'Violin',
 'Digital clock',
 'Sunflower',
 'Leopard',
 'Bell pepper',
 'Harbor seal',
 'Snake',
 'Sewing machine',
 'Goose',
 'Seat belt',
 'Coffee cup',
 'Microwave oven',
 'Countertop',
 'Serving tray',
 'Dog bed',
 'Beer',
 'Sunglasses',
 'Waffle',
 'Palm tree',
 'Trumpet',
 'Ruler',
 'Office building',
 'Pomegranate',
 'Skirt',
 'Raven',
 'Goat',
 'Kitchen knife',
 'Salt and pepper shakers',
 'Lynx',
 'Boot',
 'Platter',
 'Swimwear',
 'Swimming pool',
 'Drinking straw',
 'Wrench',
 'Ant',
 'Human ear',
 'Headphones',
 'Fountain',
 'Bird',
 'Jeans',
 'Television',
 'Crab',
 'Home appliance',
 'Snowplow',
 'Beetle',
 'Artichoke',
 'Jet ski',
 'Stationary bicycle',
 'Human hair',
 'Brown bear',
 'Lobster',
 'Drink',
 'Saucer',
 'Insect',
 'Castle',
 'Jaguar',
 'Musical instrument',
 'Taxi',
 'Pitcher',
 'Invertebrate',
 'High heels',
 'Bust',
 'Scarf',
 'Barrel',
 'Pumpkin',
 'Frog',
 'Human face',
 'Van',
 'Swim cap',
 'Ostrich',
 'Handgun',
 'Lizard',
 'Snowmobile',
 'Light bulb',
 'Window blind',
 'Muffin',
 'Pretzel',
 'Horn',
 'Furniture',
 'Fox',
 'Convenience store',
 'Fruit',
 'Earrings',
 'Curtain',
 'Sofa bed',
 'Luggage and bags',
 'Desk',
 'Crutch',
 'Bicycle helmet',
 'Tick',
 'Canary',
 'Watch',
 'Lily',
 'Kitchen appliance',
 'Filing cabinet',
 'Aircraft',
 'Cake stand',
 'Candy',
 'Mouse1',
 'Wine',
 'Drawer',
 'Picnic basket',
 'Dice',
 'Football helmet',
 'Shorts',
 'Gondola',
 'Honeycomb',
 'Chest of drawers',
 'Land vehicle',
 'Bat',
 'Dagger',
 'Tableware',
 'Human foot',
 'Mug',
 'Alarm clock',
 'Pressure cooker',
 'Human hand',
 'Sword',
 'Miniskirt',
 'Traffic sign',
 'Girl',
 'Dinosaur',
 'Porch',
 'Human beard',
 'Submarine sandwich',
 'Screwdriver',
 'Seafood',
 'Racket',
 'Wheel',
 'Toy',
 'Tea',
 'Waste container',
 'Mule',
 'Pineapple',
 'Coffee table',
 'Snowman',
 'Lavender',
 'Maple',
 'Cowboy hat',
 'Goggles',
 'Caterpillar',
 'Poster',
 'Rocket',
 'Organ',
 'Cocktail',
 'Plastic bag',
 'Mushroom',
 'Light switch',
 'Parachute',
 'Winter melon',
 'Plumbing fixture',
 'Scoreboard',
 'Envelope',
 'Bow and arrow',
 'Telephone',
 'Jacket',
 'Boy',
 'Otter',
 'Office supplies',
 'Couch',
 'Bull',
 'Whale',
 'Shirt',
 'Tank',
 'Accordion',
 'Owl',
 'Porcupine',
 'Sun hat',
 'Nail',
 'Lamp',
 'Crown',
 'Piano',
 'Sculpture',
 'Cheetah',
 'Oboe',
 'Tin can',
 'Mango',
 'Tripod',
 'Oven',
 'Coffee',
 'Common fig',
 'Salad',
 'Marine invertebrates',
 'Kangaroo',
 'Human arm',
 'Measuring cup',
 'Snail',
 'Suit',
 'Teapot',
 'Bottle',
 'Trousers',
 'Popcorn',
 'Centipede',
 'Spider',
 'Sparrow',
 'Plate',
 'Bagel',
 'Personal care',
 'Brassiere',
 'Bathroom cabinet',
 'studio couch',
 'Cabinetry',
 'Towel',
 'Nightstand',
 'Jug',
 'Wok',
 'Human eye',
 'Skyscraper',
 'Potato',
 'Paper towel',
 'Lifejacket',
 'Bicycle wheel',
 'Toilet',
 'construction--flat--crosswalk-plain',
 'human--rider--bicyclist',
 'human--rider--motorcyclist',
 'human--rider--other-rider',
 'marking--crosswalk-zebra',
 'object--banner',
 'object--bike-rack',
 'object--catch-basin',
 'object--cctv-camera',
 'object--junction-box',
 'object--mailbox',
 'object--manhole',
 'object--phone-booth',
 'object--street-light',
 'object--support--pole',
 'object--support--traffic-sign-frame',
 'object--support--utility-pole',
 'object--traffic-sign--back',
 'object--vehicle--boat',
 'object--vehicle--caravan',
 'object--vehicle--trailer',
]

unidet_mapping = {
    "cutting board" : "wood",
    "jeans" : "jean",
    "shorts" : "short",
    "tablet computer" : "tablet",
    "towel/napkin" : "napkin",
    "tape measure/ruler" : "tape measure",
    "pot/pan" : "pan",
    "frying pan" : "pan",
    "light switch" : "switch",
    "door handle" : "handle",
    "adhesive tape" : "tape",
    "bar soap" : "soap",
    "facial mask" : "facemask",
    "cigar" : "cigarette",
    "power plugs and sockets" : "plug",
    "picture frame" : "canvas",
    "chopsticks" : "chopstick",
    "mouse1" : "mouse",
    "mouse2" : "mouse",
    "green beans" : "bean",
    "computer keyboard" : "keyboard",
    "bench1" : "bench",
    "bench2" : "bench",
    "houseplant" : "plant",
    "flowerpot" : "pot",
    "game board" : "card board",
    "electric drill" : "drill",
    "cleaning products" : "vacuum cleaner",
    "salt and pepper shakers" : "seasoning",
    "human hand" : "hand",
    "card board" : "cardboard",
    "serving tray" : "tray",
    "object--support--pole" : "pole",
    "object--support--utility-pole" : "pole",
    "toothpaste" : "paste",
    "object--bike-rack" : "rack",
    "human foot" : "foot",
    "human leg" : "leg",
    "object--manhole" : "hole",
    "ceiling fan" : "fan",
    "mechanical fan" : "fan",
    "human mouth" : "mouth",
    "human arm" : "arm",
    "human head" : "head",
    "hamimelon" : "melon",
    "mobile phone" : "phone",
    "footwear" : "shoe",
    "chest of drawers" : "drawer",
    "bracelet" : "lace",
    "trash bin/can" : "dustbin",
    "furniture" : "container",
    "land vehicle" : "car",
    "fruit" : "grapefruit",
    "plastic bag" : "bag",
    "briefcase" : "bag",
    "head phone" : "headphone",

}
unidet_mapping.update(obj365_mapping)