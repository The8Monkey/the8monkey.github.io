import xml.etree.ElementTree as ET

# import os

# verzeichnis = './EA3/rawdata/'
# dateien = os.listdir(verzeichnis)

# for datei in dateien:
#     print(datei)

tree = ET.parse('./EA3/rawdata/21011.xml')
root = tree.getroot()

# with open('./EA3/processed_data/ausgabe27_06_2025.txt', 'w', encoding='utf-8') as f:
#     for rede in root.iter('rede'):
#         if rede.text:
#             f.write(rede.text.strip() + '\n')

with open('./EA3/processed_data/ausgabe06_06_2025.txt', 'w', encoding='utf-8') as f:
    for rede in root.iter('rede'):
        for elem in rede.iter():
            if len(list(elem)) == 0 and elem.text:
                f.write(elem.text.strip() + '\n')
                # print(f"{elem.text.strip()}")