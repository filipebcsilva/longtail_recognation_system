class ListIterator:
    def __init__(self, lista):
        self.lista = lista
        self.indice = 0

    def get_next(self):
        if self.indice < len(self.lista):
            item = self.lista[self.indice]
            self.indice += 1
            return item
        return None 

    def size(self):
        return len(self.lista)
