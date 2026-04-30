import pioneer


class MyNode(pioneer.TrackedNode):
    def forward(self, x):
        return x


a = MyNode()
print(a(10))
