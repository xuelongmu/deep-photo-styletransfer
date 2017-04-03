import argparse
import glob
import os
import shutil
import multiprocessing
import math
import subprocess
import scipy.misc as spm
import scipy.ndimage as spi
import scipy.sparse as sps
import numpy as np


def deep_photo(image_list, image_dir, style_dir, in_seg_dir, style_seg_dir, lap_dir,
               tmp_results_dir, results_dir, width, num_gpus, stage_1_iter,
               stage_2_iter, optimiser, λ, f_radius, f_edge):
    num_imgs = len(image_list)
    n = int(math.ceil(float(num_imgs) / num_gpus))
    processes = [None] * num_gpus
    for j in range(num_gpus):
        cmd = ''

        for i in range(n):
            idx = i * num_gpus + j
            if idx <= num_imgs:
                image_name = image_list[idx]
                image = os.path.join(image_dir, image_name)
                style_image = os.path.join(style_dir, image_name)
                in_seg_image = os.path.join(in_seg_dir, image_name)
                style_seg_image = os.path.join(style_seg_dir, image_name)
                laplacian_csv = os.path.join(lap_dir, image_name.replace(".png", "") + "_" + str(width) + ".csv")
                print('working on ' + image_name)

                part1_cmd = ' th neuralstyle_seg.lua -backend cudnn -cudnn_autotune -optimizer '+optimiser+' -content_image ' + image + ' -style_image ' + style_image + ' -content_seg ' + in_seg_image + ' -style_seg ' + style_seg_image + ' -index ' + str(
                    idx) + ' -num_iterations ' + str(
                    stage_1_iter) + ' -save_iter 100 -print_iter 1 -gpu ' + str(
                    j) + ' -serial ' + tmp_results_dir + ' &&'

                part2_cmd = ' th deepmatting_seg.lua -backend cudnn -cudnn_autotune -optimizer '+optimiser+' -content_image ' + image + ' -style_image ' + style_image + ' -init_image ' + os.path.join(
                    tmp_results_dir, "out" + str(idx) + "_t_" + str(
                        stage_1_iter) + ".png") + ' -laplacian ' + laplacian_csv + ' -content_seg ' + in_seg_image + ' -style_seg ' + style_seg_image + ' -index ' + str(
                    idx) + ' -num_iterations ' + str(
                    stage_2_iter) + ' -save_iter 100 -print_iter 1 -gpu ' + str( j) + ' -serial ' + results_dir + ' -f_radius ' + str(f_radius) + ' -f_edge ' + str(f_edge) + ' ' + '-lambda ' + str(λ) + '&&'

                cmd = cmd + part1_cmd + part2_cmd

        cmd = cmd[1:len(cmd) - 2]
        print(cmd)
        processes[j] = subprocess.Popen(cmd, shell=True)

    for j in range(num_gpus):
        processes[j].wait()
        for i in range(n):
            idx = i * num_gpus + j
            if idx <= num_imgs:
                for f in glob.iglob(os.path.join(tmp_results_dir, "out" + str(idx) +"*")):
                    os.rename(f, f.replace("out" + str(idx), image_list[idx].replace(".png","")))
                for f in glob.iglob(os.path.join(results_dir, "best" + str(idx) + "*")):
                    os.rename(f, f.replace("best" + str(idx), image_list[idx].replace(".png","")))


def getlaplacian1(i_arr: np.ndarray, consts: np.ndarray, epsilon: float = 0.0000001, win_size: int = 1):
    neb_size = (win_size * 2 + 1) ** 2
    h, w, c = i_arr.shape
    img_size = w * h
    consts = spi.morphology.grey_erosion(consts, footprint=np.ones(shape=(win_size * 2 + 1, win_size * 2 + 1)))

    indsM = np.reshape(np.array(range(img_size)), newshape=(h, w), order='F')
    tlen = int((-consts[win_size:-win_size, win_size:-win_size] + 1).sum() * (neb_size ** 2))

    row_inds = np.zeros(tlen)
    col_inds = np.zeros(tlen)
    vals = np.zeros(tlen)
    l = 0
    for j in range(win_size, w - win_size):
        for i in range(win_size, h - win_size):
            if consts[i, j]:
                continue
            win_inds = indsM[i - win_size:i + win_size + 1, j - win_size: j + win_size + 1]
            win_inds = win_inds.ravel(order='F')
            win_i = i_arr[i - win_size:i + win_size + 1, j - win_size: j + win_size + 1, :]
            win_i = win_i.reshape((neb_size, c), order='F')
            win_mu = np.mean(win_i, axis=0).reshape(1, win_size * 2 + 1)
            win_var = np.linalg.inv(
                np.matmul(win_i.T, win_i) / neb_size - np.matmul(win_mu.T, win_mu) + epsilon / neb_size * np.identity(
                    c))

            win_i2 = win_i - win_mu
            tvals = (1 + np.matmul(np.matmul(win_i2, win_var), win_i2.T)) / neb_size

            ind_mat = np.broadcast_to(win_inds, (neb_size, neb_size))
            row_inds[l: (neb_size ** 2 + l)] = ind_mat.ravel(order='C')
            col_inds[l: neb_size ** 2 + l] = ind_mat.ravel(order='F')
            vals[l: neb_size ** 2 + l] = tvals.ravel(order='F')
            l += neb_size ** 2

    vals = vals.ravel(order='F')
    row_inds = row_inds.ravel(order='F')
    col_inds = col_inds.ravel(order='F')
    a_sparse = sps.csr_matrix((vals, (row_inds, col_inds)), shape=(img_size, img_size))

    sum_a = a_sparse.sum(axis=1).T.tolist()[0]
    a_sparse = sps.diags([sum_a], [0], shape=(img_size, img_size)) - a_sparse

    return a_sparse


def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    return (im.astype('float') - min_val) / (max_val - min_val)


def reshape_img(in_img, l=512):
    in_h, in_w, _ = in_img.shape
    if in_h > in_w:
        h2 = l
        w2 = int(in_w * h2 / in_h)
    else:
        w2 = l
        h2 = int(in_h * w2 / in_w)

    return spm.imresize(in_img, (h2, w2))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-in_dir", "--in_directory", help="Path to inputs", required=True)
    parser.add_argument("-style_dir", "--style_directory", help="Path to styles", required=True)
    parser.add_argument("-in_seg_dir", "--in_seg_directory", help="Path to input segmentation", required=True)
    parser.add_argument("-style_seg_dir", "--style_seg_directory", help="Path to style segmentation", required=True)
    parser.add_argument("-tmp_results_dir", "--temporary_results_directory",
            help="Path to temporary results directory", required=True)
    parser.add_argument("-results_dir", "--results_directory", help="Path to results directory", required=True)
    parser.add_argument("-lap_dir", "--laplacian_directory", help="Path to laplacians", required=True)
    parser.add_argument("-width", "--width", help="Image width", default=512)
    parser.add_argument("-gpus", "--num_gpus", help="Number of GPUs", default=1)
    parser.add_argument("-opt", "--optimiser", help="Name of optimiser (lbfgs or adam)", default="lbfgs", choices=["lbfgs", "adam"])
    parser.add_argument("-stage_1_iter", "--stage_1_iterations", help="Iterations in stage 1", default=1000)
    parser.add_argument("-stage_2_iter", "--stage_2_iterations", help="Iterations in stage 2", default=1000)
    parser.add_argument("-lambda", dest="λ", help="Lambda parameter", type=int,
            default=10000, required=False)
    parser.add_argument("-f_radius", help="f-radius parameter", type=int,
            default=15, required=False)
    parser.add_argument("-f_edge", help="f-edge parameter", type=float,
            default=0.01, required=False)
    args = parser.parse_args()

    width = int(args.width)
    gpus = int(args.num_gpus)
    s1_iter = int(args.stage_1_iterations)
    s2_iter = int(args.stage_2_iterations)

    if not os.path.exists("/tmp/deep_photo/"):
        os.makedirs("/tmp/deep_photo/")

    if not os.path.exists("/tmp/deep_photo/in"):
        os.makedirs("/tmp/deep_photo/in")

    if not os.path.exists("/tmp/deep_photo/style"):
        os.makedirs("/tmp/deep_photo/style")

    if not os.path.exists("/tmp/deep_photo/in_seg"):
        os.makedirs("/tmp/deep_photo/in_seg")

    if not os.path.exists("/tmp/deep_photo/style_seg"):
        os.makedirs("/tmp/deep_photo/style_seg")

    if not os.path.exists(args.laplacian_directory):
        os.makedirs(args.laplacian_directory)

    if not os.path.exists(args.temporary_results_directory):
        os.makedirs(args.temporary_results_directory)

    if not os.path.exists(args.results_directory):
        os.makedirs(args.results_directory)

    files = []
    for f in glob.iglob(os.path.join(args.in_directory, '*.png')):
        files.append(f)

    good_images = []
    for f in files:
        image_name = os.path.basename(f)
        style_name = os.path.join(args.style_directory, image_name)
        image_seg_name = os.path.join(args.in_seg_directory, image_name)
        style_seg_name = os.path.join(args.style_seg_directory, image_name)
        if os.path.exists(style_name) and \
                os.path.exists(image_seg_name) and \
                os.path.exists(style_seg_name):
            good_images.append(image_name)
        else:
            print("Skipped " + str(image_name) + " due to missing files.")


    def process_image(image_name):
        filename = os.path.join(args.in_directory, image_name)
        style_name = os.path.join(args.style_directory, image_name)
        image_seg_name = os.path.join(args.in_seg_directory, image_name)
        style_seg_name = os.path.join(args.style_seg_directory, image_name)
        lap_name = os.path.join(args.laplacian_directory,
                                image_name.replace(".png", "") + "_" + str(args.width) + ".csv")

        img = spi.imread(filename, mode="RGB")
        resized_img = reshape_img(img, width)
        spm.imsave("/tmp/deep_photo/in/" + image_name, resized_img)

        style_img = spi.imread(style_name, mode="RGB")
        resized_style_img = reshape_img(style_img, width)
        spm.imsave("/tmp/deep_photo/style/" + image_name, resized_style_img)

        in_seg_img = spi.imread(image_seg_name, mode="RGB")
        resized_in_seg_img = reshape_img(in_seg_img, width)
        spm.imsave("/tmp/deep_photo/in_seg/" + image_name, resized_in_seg_img)

        style_seg_img = spi.imread(style_seg_name, mode="RGB")
        resized_style_seg_img = reshape_img(style_seg_img, width)
        spm.imsave("/tmp/deep_photo/style_seg/" + image_name, resized_style_seg_img)

        if not os.path.exists(lap_name):
            print("Calculating matting laplacian for " + str(image_name) + "...")
            img = im2double(resized_img)
            h, w, c = img.shape
            csr = getlaplacian1(img, np.zeros(shape=(h, w)), 1e-7, 1)
            coo = csr.tocoo()
            zipped = zip(coo.row + 1, coo.col + 1, coo.data)
            with open(lap_name, 'w') as out_file:
                out_file.write(str(len(coo.data))+"\n")
                for row, col, val in zipped:
                    out_file.write("%d,%d,%.15f\n" % (row, col, val))


    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(process_image, good_images)

    deep_photo(good_images, "/tmp/deep_photo/in/", "/tmp/deep_photo/style/", "/tmp/deep_photo/in_seg/",
               "/tmp/deep_photo/style_seg/", args.laplacian_directory, args.temporary_results_directory,
               args.results_directory,
               width, gpus, s1_iter, s2_iter, args.optimiser, args.λ)

    shutil.rmtree("/tmp/deep_photo/", ignore_errors=True)
