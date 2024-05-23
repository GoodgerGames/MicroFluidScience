fig, axs = plt.subplots(1, 5)

    axs[0].plot(cheb_Fx(rF, N1), y, label="r")
    axs[1].plot(F, y, label="F")
    axs[1].plot(EE, y, label="E")
    axs[3].plot(r, y, label="r after")
    axs[4].plot(K, y, label="K after")
    axs[0].set_title("t = " + str(t))
    axs[2].plot(ca, y, label="ca")


    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()
    axs[4].legend()

    plt.show()
    plt.close()