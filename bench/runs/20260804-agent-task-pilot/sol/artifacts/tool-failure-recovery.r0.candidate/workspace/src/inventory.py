class Inventory:
    def __init__(self, stock: int, reserved: int) -> None:
        self.stock = stock
        self.reserved = reserved

    def available(self) -> int:
        return max(self.stock - self.reserved, 0)
