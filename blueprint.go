package world3D

type BlockID  uint8
type Angle    struct {X, Y, Z float64}
type Space    [][][]BlockID
type AngSpace [][][]Angle

type Model interface {
    GetDim()   (uint, uint, uint)
    GetRotM()   AngSpace
    GetSpace() Space
    GetName()  string
    Valid()    bool
}

type Blueprint struct {
    name       string
    space      Space     // Region's always have a 3-dimensional space
    rotation   AngSpace  // Region's always have a rotation value for each cell in space
    dx, dy, dz uint8     // Region's always have a size
}

func NewBlueprint(name string, v Space, r AngSpace) (*Blueprint, *Error) {
    dx, dy, dz = Shape(v)
    if dx > 255 || dy > 255 || dz > 255 {
        return nil, &Error{SHAPE_ERROR, "Space too big."}
    }
    return Blueprint{name: name,
                     space: v,
                     rotation: r,
                     dx: uint8(dx),
                     dy: uint8(dy),
                     dz: uint8(dz)}, nil
}

func (bp Blueprint) GetDim() (uint, uint, uint) {return uint(bp.dx), uint(bp.dy), uint(bp.dz)}
func (bp Blueprint) GetRotM() AngSpace {return AngSpace(Copy3D(bp.rotation))
func (bp Blueprint) GetSpace() Space {return Space(Copy3D(bp.space))}
func (bp Blueprint) GetName() string {return bp.name}

func (bp Blueprint) Valid(x, y, z uint) bool {
    dx, dy, dz = bp.GetDim()
    x_valid := (x >= 0) && (x <= dx)
    y_valid := (y >= 0) && (y <= dy)
    z_valid := (z >= 0) && (z <= dz)
    return x_valid && y_valid && z_valid
}