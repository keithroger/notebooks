# %%
'''
# Machine Learning Model Comparison

This notebook will work on a mushroom data set to view how different machine
learning models perform with categorical data.

Data set availible from https://archive.ics.uci.edu/ml/datasets/Mushroom
'''

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
'''
## Column Information
'''

# %%
''''
| class| cap-shape | cap-surface | cap-color | bruises? | odor | gill-attachment | gill-spacing | gill-size | gill-color | stalk-shape | stalk-root | stalk-surface-above-ring | stalk-surface-below-ring |stalk-color-above-ring | stalk-color-below-ring | veil-type | veil-color | ring-number | ring_type | spore-rint-color | population | habitat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| edible=e |bell=b |fibrous=f |brown=n |bruises=t |almond=a |attached=a |close=c |broad=b |black=k |enlarging=e |bulbous=b |fibrous=f |fibrous=f |brown=n |brown=n |partial=p |brown=n |none=n |cobwebby=c |black=k |abundant=a |grasses=g |
| poisonous=p |conical=c |grooves=g |buff=b |no=f |anise=l |descending=d |crowded=w |narrow=n |brown=n |tapering=t |club=c |scaly=y |scaly=y |buff=b |buff=b |universal=u |orange=o |one=o |evanescent=e |brown=n |clustered=c |leaves=l |
|   |convex=x |scaly=y |cinnamon=c |  |creosote=c |free=f |distant=d |  |buff=b |  |cup=u |silky=k |silky=k |cinnamon=c |cinnamon=c |  |white=w |two=t |flaring=f |buff=b |numerous=n |meadows=m |
|   |flat=f |smooth=s |gray=g |  |fishy=y |notched=n |  |  |chocolate=h |  |equal=e |smooth=s |smooth=s |gray=g |gray=g |  |yellow=y |  |large=l |chocolate=h |scattered=s |paths=p |
|   |knobbed=k |  |green=r |  |foul=f |  |  |  |gray=g |  |rhizomorphs=z |  |  |orange=o |orange=o |  |  |  |none=n |green=r |several=v |urban=u |
|   |sunken=s |  |pink=p |  |musty=m |  |  |  |green=r |  |rooted=r |  |  |pink=p |pink=p |  |  |  |pendant=p |orange=o |solitary=y |waste=w |
|   |  |  |purple=u |  |none=n |  |  |  |orange=o |  |missing=? |  |  |red=e |red=e |  |  |  |sheathing=s |purple=u |  |woods=d |
|   |  |  |red=e |  |pungent=p |  |  |  |pink=p |  |  |  |  |white=w |white=w |  |  |  |zone=z |white=w |  |  |
|   |  |  |white=w |  |spicy=s |  |  |  |purple=u |  |  |  |  |yellow=y |yellow=y |  |  |  |  |yellow=y |  |  |
|   |  |  |yellow=y |  |  |  |  |  |red=e |  |  |  |  |  |  |  |  |  |  |  |  |  |
|   |  |  |  |  |  |  |  |  |white=w |  |  |  |  |  |  |  |  |  |  |  |  |  |
|   |  |  |  |  |  |  |  |  |yellow=y |  |  |  |  |  |  |  |  |  |  |  |  |  |
'''

# %%
column_names = [
        'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'oder',
        'gill-attatchment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
        'stalk-surface-below-ring','stalk-color-above-ring',
        'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
        'ring-type', 'spore-print-color', 'population', 'habitat']
df = pd.read_csv('data/agaricus-lepiota.data', names=column_names)

