from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from qem.crystal_analyzer import CrystalAnalyzer
from matplotlib.path import Path
from qem.atomic_column import AtomicColumns
import matplotlib.pyplot as plt
from qem.color import get_unique_colors
from ase.visualize import view
import logging
from qem.gui_classes import GetRegionSelection


@dataclass
class Region:
    name: str= None
    index: int = None
    path: Path = None
    image_shape: tuple[int, int] = None
    analyzer: Optional[CrystalAnalyzer] = None
    columns: Optional[AtomicColumns] = None

    def __str__(self):
        return f"Region data: {self.name}"

    def __repr__(self):
        return f"Region data: {self.name}"

    def view_3d(self):
        view(self.columns.lattice)
    
    @property
    def lattice(self):
        return self.columns.lattice


@dataclass
class Regions:
    region_dict: dict[int, Region] = field(default_factory=dict)
    image: np.ndarray = None

    def __init__(self, image: np.ndarray):
        self.image = image
        init_region_path = Path([
            (1, 1),
            (1, self.image.shape[1] - 2),
            (self.image.shape[0] - 2, self.image.shape[1] - 2),
            (self.image.shape[0] - 2, 1),
        ])
        self.region_dict = {0: Region(name="init", index=0, path=init_region_path, image_shape=image.shape)}

    def add_region(self, region: Region):
        self.region_dict[region.index] = region

    def remove_region(self, region: Region):
        if region.index in self.region_dict:
            del self.region_dict[region.index]
    
    def get_region(self, idx: int) -> Region:
        return self.region_dict.get(idx)
    
    def reset_regions(self):
        self.region_dict = {}

    def plot_regions(self):
        plt.imshow(self.image, cmap='gray')
        for idx, region in self.region_dict.items():
            color = get_unique_colors()
            if region.path is not None:
                path = region.path
                verts = np.array(path.vertices)
                plt.plot(verts[:, 0], verts[:, 1], label=f'Region {idx}', color=color)
                plt.fill(verts[:, 0], verts[:, 1], alpha=0.2, color=color)
        plt.legend()
        plt.show()

    @property
    def region_map(self):
        # # generate region map from regions
        region_map = np.zeros(self.image.shape, dtype=int)
        for region in self.region_dict.values():
            if region.path is not None:
                region_map[region.path.contains_points(np.indices(self.image.shape).T.reshape(-1, 2)).reshape(self.image.shape)] = region.index
        return region_map
    
    def plot(self):
        plt.figure()
        plt.imshow(self.image, cmap="gray")
        plt.imshow(self.region_map, alpha=0.5)
        plt.axis("off")
        cbar = plt.colorbar()
        cbar.set_ticks(np.arange(self.num_regions))
        plt.title("Region Map")
        plt.show()

    def __str__(self):
        return str(self.region_dict)

    def __repr__(self):
        return str(self.region_dict)

    @property
    def keys(self):
        return self.region_dict.keys()

    @property
    def values(self):
        return self.region_dict.values()

    @property
    def items(self):
        return self.region_dict.items()

    def lattice(self, index: int = None):
        if index is None:
            # combine all the regional atomic columns
            for region_index in self.region_dict.keys():
                atomic_column = self.region_dict[region_index].columns
                lattice = atomic_column.lattice
                if region_index == list(self.region_dict.keys())[0]:
                    lattice_total = lattice
                else:
                    lattice_total += lattice
            return lattice_total
        else:
            assert index in self.region_dict.keys(), "The region index is not in the regions."
            return self.region_dict[index].columns.lattice
    
    def view_3d(self, index: int = None):
        view(self.lattice(index))

    @property
    def num_regions(self):
        return len(self.region_dict.keys())
    
    def __getitem__(self, idx):
        return self.region_dict[idx]

    def draw_region(
        self, region_index: int = 0, invert_selection: bool = False
    ):
        """
        Draw a region with a polygonal selection and update the regions.

        Args:
            region_index (int, optional): The region index. Defaults to 0.
            invert_selection (bool, optional): Whether to invert the selection. Defaults to False.

        Returns:
            np.ndarray: The region mask.
        """
        atom_select = GetRegionSelection(
            image=self.image,
            invert_selection=invert_selection,
            region_map=self.region_map,
        )
        if region_index in self.keys:
            try:
                atom_select.poly.verts = self.region_dict[region_index].path.vertices
                atom_select.path = self.region_dict[region_index].path
            except KeyError:
                pass
        while plt.fignum_exists(atom_select.fig.number):
            plt.pause(0.1)

        try:
            if region_index in self.keys:
                self.region_dict[region_index].path = atom_select.path
            else:
                self.add_region(Region(
                    name=f"region_{region_index}",
                    index=region_index,
                    path=atom_select.path,
                    image_shape=self.image.shape,
                ))
        except AttributeError:
            pass
        logging.info(
            f"Assigned label {region_index} with {atom_select.region_mask.sum()} pixels to the region map."
        )
        return self.get_region(region_index)