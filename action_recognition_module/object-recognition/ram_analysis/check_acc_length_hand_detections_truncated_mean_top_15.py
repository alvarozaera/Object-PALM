import os
import json
import torch
import collections
import statistics
from tqdm import tqdm


ego4d_path = "/cluster/scratch/azaera/ego4d/v2/"
split = "val"

max_actions = 20000

with open(f'{ego4d_path}/annotations/fho_lta_taxonomy.json', 'r') as f:
    tax = json.load(f)

with open(f"{ego4d_path}/annotations/fho_lta_{split}.json", "r") as f:
    dset = json.load(f)

annotations = collections.defaultdict(list)
for entry in dset["clips"]:
    annotations[entry['clip_uid']].append(entry)

# Sort windows by their PNR frame (windows can overlap, but PNR is distinct)
annotations = {
    clip_uid: sorted(annotations[clip_uid], key=lambda x: x['action_idx'])
    for clip_uid in annotations
}


taxonomy_path = ego4d_path + "annotations/fho_lta_taxonomy.json"
with open(taxonomy_path, "r") as f:
    taxonomy = json.load(f)

nouns = taxonomy["nouns"]

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


ego4d_noun_ids_to_ram_ids = {}
for i in range(521):
    # bat (sports) is 19 and bat (tool) is 20
    # chip (food) is 84 and chip (wood metal) is 85 86
    # nut (food) is 270 and nut (tool) is 271
    # pot is 319 and pot (planter) is 320
    # vacuum cleaner is 445 and vacuum is 446
    if i == 20:
        ego4d_noun_ids_to_ram_ids[i] = ego4d_noun_ids_to_ram_ids[19]
        continue
    if i == 85 or i == 86:
        ego4d_noun_ids_to_ram_ids[i] = ego4d_noun_ids_to_ram_ids[84]
        continue
    if i == 271:
        ego4d_noun_ids_to_ram_ids[i] = ego4d_noun_ids_to_ram_ids[270]
        continue
    if i == 320:
        ego4d_noun_ids_to_ram_ids[i] = ego4d_noun_ids_to_ram_ids[319]
        continue
    if i == 446:
        ego4d_noun_ids_to_ram_ids[i] = ego4d_noun_ids_to_ram_ids[445]
        continue
    lst = []
    for tag in ram_plus_tag_labels:
        if label_mapping_ids[tag] == i:
            lst.append(tag)
    ego4d_noun_ids_to_ram_ids[i] = lst


cls_noun_tokens_path = "/cluster/project/cvg/students/azaera/noun_egovlp_distilbert_embs_cls"

nouns_sim = []
embed_nouns = []


recalls = []
total_total = 0
total_correct = 0
total_labels_len = 0

# len_indices = []
for noun_label in range(521):
    correct = 0
    total = 0
    labels_len = 0

    # Select (clip_uid, action_idx) pairs to run that have the noun_label
    to_run = []
    for clip_uid in annotations:
        for action in annotations[clip_uid]:
            if action['noun_label'] == noun_label:
                to_run.append((clip_uid, action['action_idx']))
            if len(to_run) >= max_actions:
                break
        if len(to_run) >= max_actions:
            break

    if len(to_run) == 0:
        print(f"No actions found for noun {noun_label}")
        continue

    for clip_uid, action_idx in tqdm(to_run):
        if clip_uid == '440656ae-cb82-464e-b320-25c8e693ad84':
            continue
        action = annotations[clip_uid][action_idx]
        

        ram_plus_base_outputs_path = f'/cluster/project/cvg/students/azaera/ram_plus_outputs/{clip_uid}_{action_idx}/'
        # Iterate over all directories with name frame_*

        try:
            logits = torch.load(os.path.join(ram_plus_base_outputs_path, 'aggregated_logits_hand_boxes_mapped_ego4d.pt')) #trunc-mean-3_logits_hand_boxes_max-per-frame_mapped_ego4d.pt
            logits.requires_grad = False
        except:
            total += 1
            continue

        ignore_indices = torch.tensor([2, 198, 164, 504, 230])
        logits[ignore_indices] = -float('inf')
        
        ind = torch.topk(logits, 15).indices.tolist()
        
        if action['noun_label'] in ind:
            correct += 1

        labels_len += len(ind)

        total += 1

                

    print(f"Noun {noun_label} Hit Accuracy (Recall): {correct/total} ({correct}/{total}) Average Labels Length: {labels_len/total}")
    total_correct += correct
    total_total += total
    total_labels_len += labels_len
    recalls.append(correct/total)
    

print(f"Total Hit Accuracy (Recall): {total_correct/total_total} ({total_correct}/{total_total}) Average Labels Length: {total_labels_len/total_total}")
print(f"Mean Recall: {sum(recalls)/len(recalls)}")
