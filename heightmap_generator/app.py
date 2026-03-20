import ttkbootstrap as tb

from .heightmap_generator_gui import HeightmapGeneratorGUI


def main():
    root = tb.Window(themename="darkly")
    HeightmapGeneratorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
