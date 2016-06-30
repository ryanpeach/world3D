package world3D

// Error constants:
const (
	NIL                  = iota // No error
	TYPE_ERROR           = iota // Used to signal that two parameters or a param and a value are incompatible types
	DNE_ERROR            = iota // Something Does not Exist
	ALREADY_EXISTS_ERROR = iota // Something already exists
	VALUE_ERROR          = iota // Value is not acceptable
	SHAPE_ERROR          = iota // The shape is invalid.
)

// Used to declare a general error.
// Class: Contains a simple class as declared in the const above for easy code handling.
// Info:  Also contains a detailed string for user understanding.
type Error struct {
	Class int
	Info  string
}

// Returns the size of the 3-D matrix
func Shape(s [][][]interface{}) (dx, dy, dz uint) {
    dx = len(s)
    dy = len(s[0])
    dz = len(s[0][0])
    return
}

// Returns the minimum of the integers listed
func IntMin(nums ...int) (best int) {
    best = nums[0]
    for i := 1; i < len(nums); i += 1 {
        val := nums[i]
        if val < best {
            best = val
        }
    }
    return
}

// Returns the maximum of the integers listed
func IntMax(nums ...int) (best int) {
    best = nums[0]
    for i := 1; i < len(nums); i += 1 {
        val := nums[i]
        if val > best {
            best = val
        }
    }
    return
}

// Copies a 3D slice into a new 3D slice
func Copy3D(arr [][][]interface{}) [][][]interface{} {
    dx, dy, dz = Shape(arr)
    var out [dx][dy][dz]interface{}
    for i := 0; i < dx; i += 1 {
        for j := 0; j < dy; j += 1 {
            for k := 0; k < dz; k += 1 {
                out[i][j][k] = arr[i][j][k]
            }
        }
    }
    return out[:][:][:]
}

// Copies a 3D slice into a new 3D slice
func Fill3D(val interface{}, dx, dy, dz uint) [][][]interface{} {
    dx, dy, dz = Shape(arr)
    var out [dx][dy][dz]interface{}
    for i := 0; i < dx; i += 1 {
        for j := 0; j < dy; j += 1 {
            for k := 0; k < dz; k += 1 {
                out[i][j][k] = val
            }
        }
    }
    return out[:][:][:]
}