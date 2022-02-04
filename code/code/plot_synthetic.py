import numpy as np
import pickle as pk
import matplotlib.pyplot as plt

import matplotlib.style as style
style.use('ggplot')

from matplotlib import rc
rc({'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

def plot_acc_vs_kl_800():
    d_kl = [0,0.000328,0.0267,0.151,0.240,0.430,0.744,1.002,1.484,1.863,3.136,3.198,6.189,9.826,12.243]
    dann_acc = [74.5,74.3,73.3,72.3,71.7,68.0,65.9,65.0,66.6,63.4,60.0,62.3,60.4,60.7,60.4]
    mcd_acc = [72.2,72.9,72.2,70.5,70.3,66.1,60.9,64.0,62.3,57.8,47.4,55.0,49.9,59.8,57.1]
    
    plt.plot(d_kl, dann_acc, label="DANN")
    plt.plot(d_kl, mcd_acc, label="MCD")
    plt.xlabel("$KL(p,q)$")
    plt.ylabel("Test Accuracy")
    plt.title("DANN and MCD")
    plt.legend()

    plt.savefig("plot/plots/acc_vs_kl_800.png", dpi=400)
    plt.show()

def plot_acc_vs_kl_200():
    d_kl = [0,0.000328,0.0267,0.151,0.240,0.430,0.744,1.002,1.484,1.863,3.198,3.136,6.189,9.826,12.243]
    dann_acc = [73.9,74.6,73.8,72.4,70.5,69.5,65.0,65.62,66.3,62.6,62.9,58.0,59.2,54.9,52.8]
    lr_acc = [74.80,74.48,74.42,72.29,71.49,69.16,64.11,65.03,63.79,59.92,54.57,50.50,42.53,41.06,39.49]
    pairs = sorted(list(zip(d_kl, dann_acc, lr_acc)))
    print(pairs)
    d_kl = [p[0] for p in pairs]
    dann_acc = [p[1] for p in pairs]
    lr_acc = [p[2] for p in pairs]
    
    plt.plot(d_kl, dann_acc, label="DANN")
    plt.plot(d_kl, lr_acc, label="LR")
    plt.xlabel("$KL(p,q)$")
    plt.ylabel("Test Accuracy")

    plt.legend()

    plt.savefig("plot/plots/acc_vs_kl_200.png", dpi=400)
    plt.show()

def plot_dann_acc_vs_kl():
    dann_50 = list()
    with open("plot/logs/vary_kl_data_size/dann_log_50.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            dann_50.append(tuple([float(l.strip()) for l in line.split(":")]))
    dann_100 = list()
    with open("plot/logs/vary_kl_data_size/dann_log_100.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            dann_100.append(tuple([float(l.strip()) for l in line.split(":")]))
    dann_200 = list()
    with open("plot/logs/vary_kl_data_size/dann_log_200.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            dann_200.append(tuple([float(l.strip()) for l in line.split(":")]))
    dann_400 = list()
    with open("plot/logs/vary_kl_data_size/dann_log_400.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            dann_400.append(tuple([float(l.strip()) for l in line.split(":")]))
    dann_800 = list()
    with open("plot/logs/vary_kl_data_size/dann_log_800.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            dann_800.append(tuple([float(l.strip()) for l in line.split(":")]))
    dann_2000 = list()
    with open("plot/logs/vary_kl_data_size/dann_log_2000.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            dann_2000.append(tuple([float(l.strip()) for l in line.split(":")]))
    dann_10000 = list()
    with open("plot/logs/vary_kl_data_size/dann_log_10000.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            dann_10000.append(tuple([float(l.strip()) for l in line.split(":")]))
    baseline = list()
    with open("plot/logs/vary_kl_data_size/baseline_log.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            baseline.append(tuple([float(l.strip()) for l in line.split(":")]))

    plt.plot([p[0] for p in dann_50], [p[1] for p in dann_50], label="50", color="tab:blue", alpha=0.75)
    plt.plot([p[0] for p in dann_100], [p[1] for p in dann_100], label="100", color="tab:orange", alpha=0.75)
    plt.plot([p[0] for p in dann_200], [p[1] for p in dann_200], label="200", color="tab:green", alpha=0.75)
    plt.plot([p[0] for p in dann_400], [p[1] for p in dann_400], label="400", color="tab:red", alpha=0.75)
    plt.plot([p[0] for p in dann_800], [p[1] for p in dann_800], label="800", color="tab:purple", alpha=0.75)
    plt.plot([p[0] for p in dann_2000], [p[1] for p in dann_2000], label="2000", color="tab:olive", alpha=0.75)
    plt.plot([p[0] for p in dann_10000], [p[1] for p in dann_10000], label="10000", color="tab:cyan", alpha=0.75)
    plt.plot([p[0] for p in baseline], [p[1] for p in baseline], label="baseline", color="tab:gray", alpha=0.75)

    plt.xlabel("$KL(p,q)$")
    plt.ylabel("Test Accuracy")
    plt.title("DANN with Varying Target Training Data Sizes")
    plt.legend()
    plt.savefig("plot/plots/dann_acc_vs_kl.png", dpi=400)
    plt.show()

def plot_mcd_acc_vs_kl():
    mcd_50 = list()
    with open("plot/logs/vary_kl_data_size/mcd_log_50.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            mcd_50.append(tuple([float(l.strip()) for l in line.split(":")]))
    mcd_100 = list()
    with open("plot/logs/vary_kl_data_size/mcd_log_100.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            mcd_100.append(tuple([float(l.strip()) for l in line.split(":")]))
    mcd_200 = list()
    with open("plot/logs/vary_kl_data_size/mcd_log_200.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            mcd_200.append(tuple([float(l.strip()) for l in line.split(":")]))
    mcd_400 = list()
    with open("plot/logs/vary_kl_data_size/mcd_log_400.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            mcd_400.append(tuple([float(l.strip()) for l in line.split(":")]))
    mcd_800 = list()
    with open("plot/logs/vary_kl_data_size/mcd_log_800.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            mcd_800.append(tuple([float(l.strip()) for l in line.split(":")]))
    mcd_2000 = list()
    with open("plot/logs/vary_kl_data_size/mcd_log_2000.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            mcd_2000.append(tuple([float(l.strip()) for l in line.split(":")]))
    mcd_10000 = list()
    with open("plot/logs/vary_kl_data_size/mcd_log_10000.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            mcd_10000.append(tuple([float(l.strip()) for l in line.split(":")]))
    baseline = list()
    with open("plot/logs/vary_kl_data_size/baseline_log.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            baseline.append(tuple([float(l.strip()) for l in line.split(":")]))

    plt.plot([p[0] for p in mcd_50], [p[1] for p in mcd_50], label="50", color="tab:blue", alpha=0.75)
    plt.plot([p[0] for p in mcd_100], [p[1] for p in mcd_100], label="100", color="tab:orange", alpha=0.75)
    plt.plot([p[0] for p in mcd_200], [p[1] for p in mcd_200], label="200", color="tab:green", alpha=0.75)
    plt.plot([p[0] for p in mcd_400], [p[1] for p in mcd_400], label="400", color="tab:red", alpha=0.75)
    plt.plot([p[0] for p in mcd_800], [p[1] for p in mcd_800], label="800", color="tab:purple", alpha=0.75)
    plt.plot([p[0] for p in mcd_2000], [p[1] for p in mcd_2000], label="2000", color="tab:olive", alpha=0.75)
    plt.plot([p[0] for p in mcd_10000], [p[1] for p in mcd_10000], label="10000", color="tab:cyan", alpha=0.75)
    plt.plot([p[0] for p in baseline], [p[1] for p in baseline], label="baseline", color="tab:gray", alpha=0.75)

    plt.xlabel("$KL(p,q)$")
    plt.ylabel("Test Accuracy")
    plt.title("MCD with Varying Target Training Data Sizes")
    plt.legend()
    plt.savefig("plot/plots/mcd_acc_vs_kl.png", dpi=400)
    plt.show()

def plot_lambda():
    alpha_1 = list()
    with open("plot/logs/vary_lambda/dann_log_alpha_1.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            alpha_1.append(tuple([float(l.strip()) for l in line.split(":")]))
    alpha_2 = list()
    with open("plot/logs/vary_lambda/dann_log_alpha_2.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            alpha_2.append(tuple([float(l.strip()) for l in line.split(":")]))
    alpha_4 = list()
    with open("plot/logs/vary_lambda/dann_log_alpha_4.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            alpha_4.append(tuple([float(l.strip()) for l in line.split(":")]))
    alpha_8 = list()
    with open("plot/logs/vary_lambda/dann_log_alpha_8.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            alpha_8.append(tuple([float(l.strip()) for l in line.split(":")]))
    d_kl = list()
    with open("plot/logs/vary_lambda/dann_log_d_kl.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            d_kl.append(tuple([float(l.strip()) for l in line.split(":")]))
    lambda_1 = list()
    with open("plot/logs/vary_lambda/dann_log_lambda_1.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            lambda_1.append(tuple([float(l.strip()) for l in line.split(":")]))
    lambda_2 = list()
    with open("plot/logs/vary_lambda/dann_log_lambda_2.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            lambda_2.append(tuple([float(l.strip()) for l in line.split(":")]))
    lambda_4 = list()
    with open("plot/logs/vary_lambda/dann_log_lambda_4.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            lambda_4.append(tuple([float(l.strip()) for l in line.split(":")]))
    lambda_8 = list()
    with open("plot/logs/vary_lambda/dann_log_lambda_8.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            lambda_8.append(tuple([float(l.strip()) for l in line.split(":")]))

    plt.plot([p[0] for p in alpha_1], [p[1] for p in alpha_1], label=r"$\alpha = 1$", color="tab:blue", alpha=0.75)
    plt.plot([p[0] for p in alpha_2], [p[1] for p in alpha_2], label=r"$\alpha = 2$", color="tab:orange", alpha=0.75)
    plt.plot([p[0] for p in alpha_4], [p[1] for p in alpha_4], label=r"$\alpha = 4$", color="tab:green", alpha=0.75)
    plt.plot([p[0] for p in alpha_8], [p[1] for p in alpha_8], label=r"$\alpha = 8$", color="tab:red", alpha=0.75)
    plt.plot([p[0] for p in d_kl], [p[1] for p in d_kl], label="$\lambda = KL(p,q)$", color="tab:gray", alpha=0.75)
    plt.plot([p[0] for p in lambda_1], [p[1] for p in lambda_1], label="$\lambda = 1$", color="tab:blue", linestyle="dashed", alpha=0.75)
    plt.plot([p[0] for p in lambda_2], [p[1] for p in lambda_2], label="$\lambda = 2$", color="tab:orange", linestyle="dashed", alpha=0.75)
    plt.plot([p[0] for p in lambda_4], [p[1] for p in lambda_4], label="$\lambda = 4$", color="tab:green", linestyle="dashed", alpha=0.75)
    plt.plot([p[0] for p in lambda_8], [p[1] for p in lambda_8], label="$\lambda = 8$", color="tab:red", linestyle="dashed", alpha=0.75)

    plt.xlabel("$KL(p, q)$")
    plt.ylabel("Test Accuracy")
    plt.title("DANN with Varying $\lambda$ Settings")
    plt.legend()

    plt.savefig("plot/plots/dann_lambda.png", dpi=400)
    plt.show()
    

if __name__ == "__main__":
    plot_dann_acc_vs_kl()
    plot_mcd_acc_vs_kl()
    # plot_lambda()
