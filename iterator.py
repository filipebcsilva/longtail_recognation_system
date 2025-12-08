import os
import glob

class ImageIterator:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.extensions = ['.jpg', '.jpeg', '.png', '.webp']
        
        self.image_files = []
        for ext in self.extensions:
            search_pattern = os.path.join(folder_path, f"*{ext}")
            self.image_files.extend(glob.glob(search_pattern))

        self.image_files = sorted(self.image_files)
        
        self.current_index = 0
        self.total_images = len(self.image_files)

    def get_size(self) -> int:
        return self.total_images

    def has_next(self) -> bool:
        return self.current_index < self.total_images

    def get_next(self) -> str | None:
        if self.has_next():
            image_path = self.image_files[self.current_index]
            self.current_index += 1
            return image_path
        return None
           