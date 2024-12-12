# Content of test_quickstart.mojo
from testing import assert_equal
from python import Python


struct NodeType:
    var val: String
    
    fn __init__(inout self, theType: String):
        self.val = theType


def test_execution():
    # Just tests running a mojo test
    assert_equal(0, 0)

def test_node_types():
    """
    Tests the ability to use NodeType enum
    """	
    assert_equal(0, 0)
    var nodeType = NodeType('input')
    assert_equal(nodeType.val, 'input')
