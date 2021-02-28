class Triple:

    def __init__(self, subject: str, predicate: str, object: str):
        self.subject: str = subject
        self.predicate: str = predicate
        self.object: str = object

    def __str__(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"

    def __hash__(self):
        return hash((self.subject, self.predicate, self.object))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.subject == other.subject and self.predicate == other.predicate and self.object == other.object