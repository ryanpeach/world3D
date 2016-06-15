package world3D

type BlockID uint8
type Angle struct {X, Y, Z int}
type Space [][][]BlockID
type AngSpace [][][]Angle

type Blueprint struct {
    name string
    space Space  // Region's always have a 3-dimensional space
    rotation AngSpace // Region's always have a rotation value for each cell in space
    dx, dy, dz uint      // Region's always have a size
}

func NewBlueprint(name string, V Space, R AngSpace) Blueprint {
    return Blueprint{name: name,
                     space: space,
                     rotation: rotation,
                     dx: len(space),
                     dy: len(space[0])
                     dz: len(space[0][0])}
}