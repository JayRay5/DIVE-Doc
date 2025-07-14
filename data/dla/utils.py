from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import torch.nn.functional as F



class DocLayNetSegmentationDataset(Dataset):
    def __init__(self, raws, processor,padding_index=12,return_doc_cls=False,mask_size=None):
        """
        raws: liste de dicts comme ton exemple
        model_input_size: (w, h) de ton réseau, ex. (224,224)
        patch_size: taille des patches carrés
        transform: transforms à appliquer aux patches images (torchvision)
        """
        self.raws = raws
        self.model_h,self.model_w = processor.image_processor.size["height"],processor.image_processor.size["width"]
        self.processor = processor
        self.padding_index = padding_index
        self.return_doc_cls = return_doc_cls
        self.mask_size = mask_size
    def __len__(self):
        return len(self.raws)
    
    def __compute_mask_dim__(self,processed_img):
        valid_pixels = processed_img[0, :, :] != -1 
        coords = valid_pixels.nonzero(as_tuple=False)   # shape (num_true, 2)

        if coords.size(0) == 0:
            # No True pixels
            height = width = 0
        else:
            # 2) Split into rows and cols
            rows = coords[:, 0]
            cols = coords[:, 1]

            # 3) Compute min/max for each dimension
            rmin, rmax = rows.min().item(), rows.max().item()
            cmin, cmax = cols.min().item(), cols.max().item()

            # 4) Compute height and width of the bounding box
            height = (rmax - rmin + 1)
            width  = (cmax - cmin + 1)
            return(height,width)

    def __padd_mask__(self,img_size,mask):
        pad_top = (img_size[-2] - mask.shape[-2]) // 2
        pad_bottom = img_size[-2] - mask.shape[-2] - pad_top

        pad_left = (img_size[-1] - mask.shape[-1]) // 2
        pad_right = img_size[-1] - mask.shape[-1] - pad_left
        padded_mask = F.pad(mask,(pad_left, pad_right, pad_top, pad_bottom),value=self.padding_index)
        return padded_mask
    
    def __getitem__(self, idx):
        
        raw = self.raws[idx]

        if self.processor!=None:
            processed_image = self.processor(raw["image"].convert("RGB"), return_tensors="pt").pixel_values[0]
            if self.mask_size != None:
                tensor = torch.zeros((1,self.mask_size[0],self.mask_size[1]))
                mask_dim = self.__compute_mask_dim__(tensor)
            else: 
                mask_dim = self.__compute_mask_dim__(processed_image)
              
        H, W = raw['height'], raw['width']

        # --- 1) Construire le mask 2d ---
        mask = Image.new('L', (H, W), 0)                    # niveau de gris
        draw = ImageDraw.Draw(mask)
        for obj in raw['objects']:
            # segmentation est [[x1,y1, x2,y2, ...]]
            for seg in obj['segmentation']:
                draw.polygon(seg, outline=obj["category_id"], fill=obj["category_id"])
        mask = np.array(mask, dtype=np.uint8)               # shape (H, W), valeurs 0/1

        # --- 2) Redimensionner  mask ---
        mask_resized = Image.fromarray(mask)          
        mask_resized = mask_resized.resize((mask_dim[1],mask_dim[0]), Image.NEAREST)
        mask_resized = np.array(mask_resized)
        mask_resized = torch.Tensor(mask_resized)

        # --- 3) Ajouter le padd pour matcher l'image d'origine si reconstruction sur la taille de l'image originel
        if self.mask_size==None:
            mask_resized = self.__padd_mask__([self.model_h,self.model_w],mask_resized)

        mask_resized = mask_resized.long() 

        if self.return_doc_cls:
                   return mask_resized, processed_image,raw["doc_category"]
        return mask_resized, processed_image
    
    
    
 #ne marche pas pour des batches > 1 car taille d'image unpadded peu varier

#Allow to remove padding in the outputed segmentation to keep the groundtruth area
# take input of dim (B,H,W,C)
def post_process_seg(segmentation,mask):
    H,W = segmentation.shape[len(segmentation.shape)-3],segmentation.shape[len(segmentation.shape)-2]
    h,w = (H - mask.shape[len(mask.shape)-3])//2, (W - mask.shape[len(mask.shape)-2])//2
    if len(segmentation.shape)==3:
        unpadded_segmentation = segmentation[0+h:H-h,0+w:W-w,:]
    elif len(segmentation.shape)==4:
        unpadded_segmentation = segmentation[:,0+h:H-h,0+w:W-w,:]
    else:
        raise ValueError(f"Expect inputs of 3 or 4 dim, got {len(segmentation.shape)}")
    return unpadded_segmentation

def plot_bar():
    id2label = {
                0: "Caption",
                1: "Footnote",
                2: "Formula",
                3: "List-item",
                4: "Page-footer",
                5: "Page-header",
                6: "Picture",
                7: "Section-header",
                8: "Table",
                9: "Text",
                10: "Title",
                11: "Padding"
            }
    palette_12_cls = [
                    "#8DD3C7",  # turquoise
                    "#FFFFB3",  # jaune clair
                    "#BEBADA",  # mauve
                    "#FB8072",  # saumon
                    "#80B1D3",  # bleu ciel
                    "#FDB462",  # orange clair
                    "#B3DE69",  # vert clair
                    "#FCCDE5",  # rose poudré
                    "#D9D9D9",  # gris clair
                    "#BC80BD",  # violet clair
                    "#CCEBC5",  # vert pastel
                    "#000000",  # black for the padding
                ]

    custom_cmap = ListedColormap(palette_12_cls)
    
    plt.figure(figsize=(8, 1))
    plt.imshow(np.arange(12).reshape(1, -1), cmap=custom_cmap, aspect='auto')
    plt.xticks(np.arange(12), [id2label[i] for i in range(12)], rotation=45, ha='right')
    plt.yticks([])
    plt.title("Class Color Map")
    plt.tight_layout()
    plt.show()
def plot_segmentation(image,mask,save=False,path_name=""):
    palette_12_cls = [
                    "#8DD3C7",  # turquoise
                    "#FFFFB3",  # jaune clair
                    "#BEBADA",  # mauve
                    "#FB8072",  # saumon
                    "#80B1D3",  # bleu ciel
                    "#FDB462",  # orange clair
                    "#B3DE69",  # vert clair
                    "#FCCDE5",  # rose poudré
                    "#D9D9D9",  # gris clair
                    "#BC80BD",  # violet clair
                    "#CCEBC5",  # vert pastel
                    "#000000",  # black for the padding
                ]

    custom_cmap = ListedColormap(palette_12_cls)
    if save:
        if path_name==None or path_name=="":
            raise ValueError(f"name should be specified, got {path_name}")
    plt.figure(figsize=(15, 8))
    plt.imshow(mask, cmap=custom_cmap,vmin=0,vmax=11)
    plt.imshow(image,alpha=0.1)
    if save:
        plt.savefig(path_name, format="png", dpi=300)