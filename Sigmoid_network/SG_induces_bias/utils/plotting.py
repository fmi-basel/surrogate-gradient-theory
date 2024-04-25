import matplotlib.pyplot as plt
import seaborn as sns


def contour(
    ax,
    x,
    y,
    z,
    levels=20,
    vmin=0,
    vmax=1,
    color="k",
    linewidths=0.5,
    cmap="viridis",
    alpha=1,
):
    im = ax.contourf(x, y, z, levels, vmin=vmin, vmax=vmax, cmap=cmap)
    if color == None:
        ax.contour(x, y, z, levels, cmap=cmap, linewidths=linewidths, alpha=alpha)
    else:
        ax.contour(x, y, z, levels, colors=color, linewidths=linewidths, alpha=alpha)
    return im


def surface_and_line(
    ax, X, Y, Z, xt, yt, St, xlabel="x", ylabel="y", elev=45, azim=45, cmap="cividis"
):
    ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.95)
    ax.plot(xt, yt, 0, color="gray")
    ax.plot(xt, yt, St, color="black")
    ax.view_init(elev=elev, azim=azim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("loss")

    sns.despine()
    plt.tight_layout()


def path(
    ax,
    Sd,
    St,
    color="black",
    ylabel="Loss",
    fill=True,
    linestyle="solid",
    normalize=True,
):
    if normalize:
        # normalize St between 0 and 1
        St = (St - St.min()) / (St.max() - St.min())
        ax.set_ylim(0, 1.05)

    Sd = (Sd - Sd.min()) / (Sd.max() - Sd.min())

    ax.plot(Sd, St, color=color, linestyle=linestyle)
    if fill:
        ax.fill_between(Sd, St, color=color, alpha=0.25)
    ax.set_xlabel("path")
    ax.set_ylabel(ylabel)
    ax.hlines(St[0], 0, 1, color=color, linestyle="dashed")

    sns.despine()


def line_in_2d(xt, yt, t):
    fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.5), dpi=200)
    ax[0].plot(xt, "k", linewidth=2)
    ax[0].plot(yt, "gray", linewidth=2)

    ax[1].plot(xt, yt, color="black", zorder=-5)
    ax[1].scatter(xt[0], yt[0], color="black", marker="o", s=20)
    ax[1].scatter(xt[-1], yt[-1], color="red", marker="o", s=10)

    ax[1].scatter(xt[len(t) // 2], yt[len(t) // 2], color="black", marker="o", s=10)

    sns.despine()
    plt.tight_layout()
