from matplotlib import pyplot as plt
import torch
from glob import glob
from matplotlib import font_manager
from PIL import Image 
import os
from natsort import natsorted

plt.rcParams['font.size'] = 28
def concat_images_horizontally(image_paths, output_path):
    images = [Image.open(p) for p in image_paths]

    min_height = min(img.height for img in images)
    resized_images = [
        img.resize((int(img.width * min_height / img.height), min_height), Image.LANCZOS)
        for img in images
    ]

    total_width = sum(img.width for img in resized_images)

    new_img = Image.new('RGB', (total_width, min_height))
    current_x = 0
    for img in resized_images:
        new_img.paste(img, (current_x, 0))
        current_x += img.width
    new_img.save(output_path)

def visualize_progress_and_policy(ys, lat0, lat1, lon0, lon1, dep0, dep1, tim0, tim1, initial_point, sample_point, high_score_ponit, savename, itr=1, top_number=1):
    if itr == 1 :
        fig, ax = plt.subplots(2, 1, figsize=(9, 18), dpi=300)
    elif top_number == 1 :
        fig, ax = plt.subplots(2, 1, figsize=(9.2, 18), dpi=300)
    else:
        fig, ax = plt.subplots(2, 1, figsize=(7.8, 18), dpi=300)
    plt.subplots_adjust(wspace=0.1)
    ys = ys.reshape(61, 61, 61, 81)
    max_3d, _ = torch.max(ys, dim=2)
    max_4d, _ = torch.max(max_3d, dim=2)
    c = ax[0].imshow(max_4d.cpu(), origin="lower", aspect='auto', alpha=0.9, cmap='PuBu', extent=[lon0.cpu(), lon1.cpu(), lat0.cpu(), lat1.cpu()])
    #c = ax[0].imshow(max_4d.cpu().T, origin="lower", aspect='auto',cmap='PuBu_r', extent=[lat0.cpu(), lat1.cpu(), lon0.cpu(), lon1.cpu()])
    if top_number == 1:
        plt.colorbar(c, ax=ax[0])
    if itr != 1:
        ax[0].yaxis.set_visible(False)
    ax[0].set_ylabel(r"Latitude (°)", fontsize=32)
    ax[0].set_xlabel(r"Longitude (°)", fontsize=32)
    ax[0].scatter(initial_point.cpu()[0, 1], initial_point.cpu()[0, 0], marker="^", c="red",s=200,  label='initial point', edgecolors='black')
    ax[0].text(0.02, 0.96, f'({chr(96+itr)})', transform=ax[0].transAxes, fontsize=30, va='top', ha='left')
    #ax[0].scatter(sample_point.cpu()[0, 0, ...], sample_point.cpu()[0, 1, ...],  c=sample_point.cpu()[0, 4, ...], marker="+", s=100, label='sample point')
    if itr == 1:
        ax[0].scatter(sample_point.cpu()[0, 1, ...], sample_point.cpu()[0, 0, ...],  c='black', marker="o", s=0.5 , edgecolors=None, label='samples')
    else:
        ax[0].scatter(sample_point.cpu()[0, 1, ...], sample_point.cpu()[0, 0, ...], c='black', marker="o", s=0.05,   edgecolors=None, label='samples')
    ax[0].scatter(high_score_ponit.cpu()[0, 1, ...], high_score_ponit.cpu()[0, 0, ...], marker="D", s=100, c="white", edgecolors='black', label='top-values')
    ax[0].legend(fontsize=25, loc='upper right', fancybox=True)

    max_3d, _ = torch.max(ys, dim=0)
    max_4d, _ = torch.max(max_3d, dim=0)
    d = ax[1].imshow(max_4d.cpu().T, origin="lower", aspect='auto',alpha=0.9, cmap='PuBu', extent=[dep0.cpu(), dep1.cpu(), tim0.cpu(), tim1.cpu()])
    #d = ax[1].imshow(max_4d.cpu().T, origin="lower", aspect='auto', cmap='PuBu_r', extent=[dep0.cpu(), dep1.cpu(), tim0.cpu(), tim1.cpu()])
    if top_number == 1 :
        plt.colorbar(d, ax=ax[1])
    if itr != 1:
        ax[1].yaxis.set_visible(False)
    ax[1].set_xlabel(r"Depth (km)", fontsize=32)
    ax[1].set_ylabel(r"Time (s)", fontsize=32)
    ax[1].scatter(0, initial_point.cpu()[0, 3], marker="^", c="red", s=200, label='initial point', edgecolors='black')
    ax[1].text(0.02, 0.96, f'({chr(96+itr+5)})', transform=ax[1].transAxes, fontsize=30, va='top', ha='left')
    #ax[1].scatter(sample_point.cpu()[0, 2, ...], sample_point.cpu()[0, 3, ...], c=sample_point.cpu()[0, 4, ...], marker="+", s=100, label='sample point')
    if itr != 1:
        ax[1].scatter(sample_point.cpu()[0, 2, ...], sample_point.cpu()[0, 3, ...], c='black', marker="o", s=0.05, edgecolors=None, label='samples')
    else:
        ax[1].scatter(sample_point.cpu()[0, 2, ...], sample_point.cpu()[0, 3, ...], c='black', marker="o", s=0.5,  edgecolors=None, label='samples')
    ax[1].scatter(high_score_ponit.cpu()[0, 2, ...], high_score_ponit.cpu()[0, 3, ...], marker="D",s=100, c="white", edgecolors='black', label='top-values')
    ax[1].legend(fontsize=25, loc='upper right', fancybox=True)
    plt.tight_layout(pad=0.1)
    plt.savefig(savename)
    plt.close()
    
    if top_number == 1 :
        plot_dir = savename.rsplit('/', 1)[0]
        image_paths = natsorted(glob(plot_dir + '/iter*'))
        output_path = plot_dir + '/doublef.png'
        concat_images_horizontally(image_paths, output_path)
        
