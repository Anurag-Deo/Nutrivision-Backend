from simple_image_download import simple_image_download as simp

lst = [
    "malapua",
    "chana_masala",
    "mysore_pak",
    "paneer_butter_masala",
    "anarsa",
    "qubani_ka_meetha",
    "sheer_korma",
    "bhatura",
]


response = simp.simple_image_download
for rep in lst:
    response().download(rep, 400)
    print(f"]Downloaded {rep} images.")

# "daal_baati_churma",
#     "butter_chicken",
#     "malapua",
#     "chana_masala",
#     "mysore_pak",
#     "paneer_butter_masala",
#     "anarsa",
#     "qubani_ka_meetha",
#     "sheer_korma",
#     "bhatura",
#     "poha",
#     "misti_doi",
#     "lassi",
#     "aloo_matar",
#     "pootharekulu",
#     "gajar_ka_halwa",
#     "kachori",
#     "kalakand",
#     "rabri",
#     "daal_puri",
#     "maach_jhol",
#     "dal_makhani",
#     "basundi",
#     "misi_roti",
#     "sheera",
#     "imarti",
#     "sohan_papdi",
#     "lyangcha",
#     "dum_aloo",
