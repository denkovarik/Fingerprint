struct NodeType:
    var val: String
    
    fn __init__(inout self, theType: String):
        self.val = theType