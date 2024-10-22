import os
import json
import torch
from tqdm import tqdm
from collections import Counter

split = 'train'
ego4d_path = '/cluster/scratch/azaera/ego4d/v2'

with open(f'{ego4d_path}/annotations/fho_lta_taxonomy.json', 'r') as f:
    tax = json.load(f)

with open(f'{ego4d_path}/annotations/fho_lta_{split}.json', 'r') as f:
    dset = json.load(f)

label_mapping_ids_path = '../ram_nouns_mapping_ids.json'

with open(label_mapping_ids_path, 'r') as f:
    label_mapping_ids = json.load(f)

not_originally_in_ram_nouns_path = '../ram_nouns_not_in_tax.txt'

# AUGMENT MAPPING WITH NOUNS NOT ORIGINALLY IN RAM
with open(not_originally_in_ram_nouns_path, 'r') as f:
    not_originally_in_ram_nouns = f.readlines()
not_originally_in_ram_nouns = [line.strip() for line in not_originally_in_ram_nouns]

for noun in not_originally_in_ram_nouns:
    label_used_in_ram = noun.split('_(')[0].replace('_', ' ')
    label_mapping_ids[label_used_in_ram] = tax['nouns'].index(noun)

ram_plus_tag_labels = ['acrylic paint', 'plane', 'ambulance', 'apple', 'apron', 'arm', 'art',
 'asparagus', 'assembly', 'eggplant', 'avocado', 'ax', 'baby', 'back', 'bacon',
 'bag', 'ball', 'balloon', 'bamboo', 'banana', 'bandage', 'bracelet', 'bar',
 'baseball', 'basin', 'basket', 'basketball', 'bat', 'bathroom', 'toilet paper',
 'batter', 'battery', 'bead', 'beaker', 'bean', 'bed', 'beef', 'beer',
 'beer glass', 'clock', 'belt', 'belt buckle', 'bench', 'berry', 'beverage',
 'bicycle', 'bin', 'bird', 'biscuit', 'blanket', 'blender', 'block', 'paddle',
 'bolt', 'book', 'bookcase', 'boot', 'bottle', 'bowl', 'box', 'boxer', 'brake',
 'branch', 'bread', 'break', 'brick', 'wall', 'broccoli', 'broom', 'brownie',
 'brush', 'bubble gum', 'bud', 'lamp', 'butter', 'cream', 'butterfly', 'button',
 'cabbage', 'cabinet', 'cable', 'cake', 'calculator', 'camera', 'can',
 'can opener', 'candle', 'jar', 'canvas', 'cap', 'car', 'card', 'cardboard',
 'cardigan', 'carpet', 'slipper', 'carrot', 'cart', 'carton', 'case', 'cash',
 'cat', 'ceiling', 'celery', 'cello', 'smartphone', 'cement', 'cereal', 'chain',
 'chair', 'chalk', 'champagne', 'charger', 'cheese', 'cheeseburger', 'chess',
 'chicken', 'chip', 'chisel', 'chocolate', 'chocolate chip cookie',
 'cutting board', 'chopstick', 'cigarette', 'circuit', 'clay', 'clip', 'cloth',
 'coat', 'cocoa', 'coconut', 'coffee', 'coffee machine', 'coin', 'colander',
 'comb', 'computer', 'keyboard', 'concrete', 'connector', 'container', 'control',
 'cooker', 'rope', 'cork', 'corn', 'corner', 'couch', 'counter', 'courgette',
 'cow', 'crab', 'craft', 'crate', 'crayon', 'crochet', 'crowbar', 'cucumber',
 'cup', 'cupcake', 'curtain', 'debris', 'table', 'detergent', 'die', 'dirt',
 'plate', 'dish washer', 'dog', 'doll', 'domino', 'donut', 'door', 'dough', 'draw',
 'drawer', 'drawing', 'dress', 'drill', 'drink', 'drone', 'medicine', 'drum',
 'dumbbell', 'dust', 'dustpan', 'duvet', 'earphone', 'earplug', 'egg',
 'elastic band', 'engine', 'envelope', 'eraser', 'fabric', 'fan', 'faucet',
 'fence', 'fiber', 'file', 'filter', 'hand', 'fish', 'flask', 'floor', 'flour',
 'flower', 'horse', 'foam', 'foil', 'leaf', 'food', 'foot', 'fork', 'French fries',
 'fridge', 'fuel', 'funnel', 'game controller', 'ham', 'garbage', 'garlic',
 'garment', 'gas', 'gauge', 'gear', 'generator', 'ginger', 'glasses', 'glove',
 'golf club', 'gourd', 'grain', 'grape', 'grapefruit', 'grass', 'grater', 'grill',
 'grinder', 'guitar', 'hair', 'hamburger', 'hammer', 'handkerchief', 'handle',
 'hanger', 'hat', 'hay', 'head', 'heater', 'helmet', 'hinge', 'hoe', 'hole', 'hook',
 'hose', 'hot dog', 'house', 'person', 'ice', 'icecream', 'light', 'ink', 'ipad',
 'iron', 'jack', 'jacket', 'jug', 'juice', 'juicer', 'kale', 'key', 'kiwi', 'knife',
 'knob', 'knot', 'label', 'lace', 'ladder', 'ladle', 'laptop', 'lead', 'leash',
 'leg', 'lemon', 'lettuce', 'lid', 'lift', 'lime', 'lock', 'log', 'lumber',
 'machete', 'magazine', 'magnet', 'mallet', 'man', 'mango', 'marble', 'mask',
 'mat', 'match', 'material', 'measuring tape', 'meat', 'melon', 'net', 'metal',
 'microscope', 'microwave', 'milk', 'mirror', 'mixer', 'mixture', 'mold', 'money',
 'mortar', 'motherboard', 'motor', 'motorbike', 'motorcycle', 'mouse', 'mouth',
 'mower', 'mud', 'mug', 'mushroom', 'nail polish', 'napkin', 'necklace', 'needle',
 'noodle', 'note', 'notebook', 'nut', 'oil', 'okra', 'onion', 'orange', 'ornament',
 'oven', 'pack', 'package', 'paint brush', 'palette', 'pan', 'pancake', 'panel',
 'pant', 'pants', 'papaya', 'paper', 'paper cutter', 'pasta', 'paste', 'pastry',
 'pea', 'peach', 'peanut', 'pear', 'peel', 'peeler', 'pen', 'pencil', 'pepper',
 'phone', 'photo', 'piano', 'pickle', 'picture', 'pie', 'tablet', 'pillow', 'pin',
 'pipe', 'pizza', 'plank', 'plant', 'platter', 'playing card', 'plier', 'plug',
 'plum', 'plywood', 'pole', 'popcorn', 'portrait', 'stamp', 'poster', 'pot',
 'potato', 'pouch', 'printer', 'pump', 'pumpkin', 'puzzle', 'racket', 'radio',
 'rail', 'rake', 'ratchet', 'razor blade', 'receipt', 'reed', 'remote',
 'restroom', 'ribbon', 'rice', 'ring', 'road', 'stone', 'coaster', 'rolling pin',
 'root', 'router', 'ruler', 'sack', 'salad', 'salt', 'sand', 'sandal', 'sandwich',
 'sauce', 'saucer', 'sausage', 'saw', 'scaffold', 'scarf', 'scissors', 'scoop',
 'scraper', 'screen', 'screw', 'screwdriver', 'sculpture', 'seal', 'seed',
 'sewing machine', 'shaker', 'sharpener', 'shawl', 'shears', 'sheet', 'shelf',
 'shell', 'shirt', 'shoe', 'short', 'shot glass', 'shoulder', 'shovel',
 'shower head', 'shrub', 'sink', 'sketch', 'snorkel', 'soap', 'sock', 'socket',
 'soup', 'spaghetti', 'wrench', 'spatula', 'speaker', 'sphere', 'spice',
 'spinach', 'sponge', 'spoon', 'spray', 'spring', 'squash', 'stair', 'stairs',
 'stand', 'stapler', 'steel', 'steering wheel', 'stem', 'stereo', 'stick',
 'sticker', 'stock', 'stool', 'stop watch', 'tank', 'stove', 'strainer', 'strap',
 'straw', 'strawberry', 'string', 'sugar', 'sweater', 'sweatshirt', 'switch',
 'syringe', 'taco', 'tape', 'tea', 'tea pot', 'television', 'tent', 'test tube',
 'thermometer', 'tie', 'tile', 'tin', 'tissue', 'toaster', 'toe', 'tomato',
 'tongs', 'toolbox', 'toothbrush', 'toothpick', 'torch', 'tortilla', 'towel',
 'toy', 'tractor', 'trolley', 'tray', 'treadmill', 'tree', 'truck', 'tub', 'twig',
 'twine', 'umbrella', 'vacuum', 'valve', 'vase', 'vegetable', 'vehicle',
 'video game', 'vine', 'vinegar', 'violin', 'wallpaper', 'washing machine',
 'waste', 'watch', 'water', 'watermelon', 'weed', 'weight scale', 'wheat',
 'wheel', 'whisk', 'window', 'wiper', 'windshield', 'wine glass', 'wire', 'woman',
 'wood', 'wool', 'worm', 'wrap', 'yoghurt', 'zip', 'zipper', 'awl', 'baking soda',
 'ball bearing', 'baseboard', 'blower', 'bolt extractor', 'brake pad', 'bucket',
 'caliper', 'chaff', 'clamp', 'cushion', 'derailleur', 'doorbell', 'dough mixer',
 'drill bit', 'duster', 'facemask', 'filler', 'fishing rod', 'flash drive',
 'gasket', 'gate', 'gauze', 'glue', 'glue gun', 'guava', 'haystack', 'ketchup',
 'kettle', 'lever', 'lighter', 'lubricant', 'manure', 'mop', 'multimeter',
 'nail cutter', 'nail gun', 'nozzle', 'paint roller', 'pedal', 'peg',
 'pilot jet', 'purse', 'rack', 'rod', 'sander', 'sandpaper', 'set square',
 'sickle', 'sketch pad', 'skirt', 'slab', 'solder iron', 'spacer',
 'sphygmomanometer', 'spirit level', 'squeezer', 'steamer', 'stroller',
 'trimmer', 'trowel', 'tweezer', 'vacuum cleaner', 'wallet', 'welding torch',
 'wheelbarrow', 'yam', 'yeast', 'zucchini', 'baton', 'cash register', 'cassava',
 'leek', 'pipette', 'plunger', 'putty', 'transistor']



#nouns_distr = {noun:[] for noun in ram_plus_tag_labels}
nouns_distr = {noun_label:[] for noun_label in range(len(tax['nouns']))}


#all_quantiles = torch.zeros(len(ram_plus_tag_labels), 5)
all_quantiles = torch.zeros(len(tax['nouns']), 15)

for action in tqdm(dset['clips']):
    try:
        clip_uid = action['clip_uid']
        action_idx = action['action_idx']
        noun = action['noun_label']
        
        ram_plus_base_outputs_path = f'/cluster/project/cvg/students/azaera/ram_plus_outputs/{clip_uid}_{action_idx}/'
        logits = torch.load(os.path.join(ram_plus_base_outputs_path, 'aggregated_logits_hand_boxes_mapped_ego4d.pt')) # 'max_logits_hand_boxes_mapped_ego4d.pt' trunc-mean-3_logits_hand_boxes_max-per-frame_mapped_ego4d.pt
        # nouns_ram_gt = [(i, label) for i, label in enumerate(ram_plus_tag_labels) if label_mapping_ids[label] == noun]
        # nouns_ram_gt_ids = [i for i, _ in nouns_ram_gt]
        # for noun_idx, noun_ram_gt in nouns_ram_gt:
        #     logit = logits[noun_idx].item()
        #     nouns_distr[noun_ram_gt].append(logit)
        
        # for i, logit in enumerate(logits):
        #     if True:#i not in nouns_ram_gt_ids:
        #         nouns_distr_not_gt[ram_plus_tag_labels[i]].append(logit.item())

        for i,logit in enumerate(logits):
            #nouns_distr[ram_plus_tag_labels[i]].append(logit.item())
            if i != noun:
                nouns_distr[i].append(logit.item())

    except Exception as e:
        print(f'Error in {clip_uid}_{action_idx}: {e}')
        continue


for noun in range(len(tax['nouns'])):
    quantiles = torch.tensor(nouns_distr[noun]).quantile(torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999]))
    all_quantiles[noun] = quantiles
    print(all_quantiles[noun])

# Save all_quantiles (torch tensor) to a file
torch.save(all_quantiles, 'non-gt_quantiles_train_ram_plus_trunc-mean-3_logits_hand_boxes_mapped_ego4d_complete.pt')
