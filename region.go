package world3D

import "math"

type LayerSpace [][][]int
type Region struct {
    bp *Blueprint
    layer LayerSpace
    x0, y0, z0 int
    rotation Angle
}

func NewRegion(bp *Blueprint, x, y, z, l int, rotation Angle) Blueprint {
    dx, dy, dz = bp.GetDim()
    L = LayerSpace(Fill3D(l, dx, dy, dz))
    return Region{bp: bp,
                  layer: L,
                  x, y, z: x, y, z,
                  rotation: rotation}
}

func (r Region) GetBP() *Blueprint {return r.bp}
func (r Region) GetLayer() *Blueprint {return LayerSpace(Copy3D(r.layer))}
func (r Region) GetLoc() (int, int, int) {return r.x0, r.y0, r.z0}
func (r Region) GetRot() Angle {return r.rotation}

func (r Region) GetDim() (uint, uint, uint) {return r.GetBP().GetDim()}
func (r Region) GetRotM() AngSpace {return r.GetBP().GetRotM()}
func (r Region) GetSpace() Space {return r.GetBP().GetSpace()}
func (r Region) GetName() string {return r.GetBP().GetName()}
func (r Region) Valid() bool {return r.GetBP().Valid()}

// Tests whether or not x, y, and z are all within this Region
func (r Region) LocValid(x, y, z int) bool {
    x0, y0, z0 = r.GetLoc()
    dx, dy, dz = r.GetDim()
    x_valid := (x >= x0 - dx/2.) && (x <= x0 + dx)
    y_valid := (y >= y0 - dy/2.) && (y <= y0 + dy)
    z_valid := (z >= z0 - dz/2.) && (z <= z0 + dz)
    return x_valid && y_valid && z_valid
}

//[wx]   [1  0  0 ] [c2  0 s2] [c3 -s3  0]  [ [1 0 0 0] [ [1 0 0 -dx/2][i] ]]T
//[wy] = [0 c1 -s1] [0   1  0] [s3  c3  0]  | [0 1 0 0] | [0 1 0 -dy/2][j] ||
//[wz]   [0 s1 c1 ] [-s2 0 c2] [0    0  1]  [ [0 0 1 0] | [0 0 1 -dz/2][k] ||
//                                                      [ [0 0 0   1. ][1] ]]
//[x]   [1, 0, 0, x0] [wx]
//[y] = [0, 1, 0, y0] [wy]
//[z] = [0, 0, 1, z0] [wz]
//[1]   [0, 0, 0, 1 ] [1]
func Coord2Loc(r Region, i, j, k uint) (x, y, z int) {
    a1, a2, a3 := r.GetRot()
    c1, c2, c3 := math.Cos(a1), math.Cos(a2), math.Cos(a3)
    s1, s2, s3 := math.Sin(a1), math.Sin(a2), math.Sin(a3)
    idx, idy, idz := r.GetDim()
    ix0, iy0, iz0 := r.GetLoc()
    x0, y0, z0 := float64(ix0), float64(iy0), float64(iz0)
    i0, j0, k0 := -float64(idx)/2.0, -float64(idy)/2.0, -float64(idz)/2.0
    fi, fj, fk := float64(i), float64(j), float64(k)
    wx := c2*c3*(fi+i0)−c2*s3*(fj+j0)+s2*(fk+k0)
    wy := −c2*s1*(fk+k0)+(fi+i0)*(c1*s3+c3*s1*s2)+(fj+j0)*(c1*c3−s1*s2*s3)
    wz := c1*c2*(fk+k0)+(fi+i0)*(−c1*c3*s2+s1*s3)+(fj+j0)*(c1*s2*s3+c3*s1)
    x, y, z = int(wx+x0), int(wy+y0), int(wz+z0)
    return
}

// Adds two regions together, creating a new region large enough for both
func AddRegions(r1 *Region, r2 *Region, name string) (out Region) {
    // Get some border info
    r1x0, r1y0, r1z0, r1x1, r1y1, r1z1 := r1.Border()
    r1x0, r1y0, r1z0, r1x1, r1y1, r1z1 := r1.Border()
    
    // Find the final corner locations, calculate the size
    x0, y0, z0 := IntMin(r1x0, r2x0), IntMin(r1y0, r2y0), IntMin(r1z0, r2z0)
    x1, y1, z1 := IntMax(r1x1, r2x1), IntMax(r1y1, r2y1), IntMax(r1z1, r2z1)
    dx, dy, dz := x1-x0, y1-y0, z1-z0

    // Create variables before hand to avoid memory handling speed
    var {
        blockM [dx][dy][dz]BlockID  // The output block space
        angM   [dx][dy][dz]Angle    // The output angle space
        layerM [dx][dy][dz]int      // The output layer space
        block BlockID               // Temporary block storage
        ang   Angle                 // Temporary angle storage
        layer int                   // Temporary layer storage
    }
    
    // Iterate over x
    for i = 0 ; i < dx; i += 1 {
        for j = 0 ; j < dy; j += 1 {
            for k = 0 ; k < dz; k += 1 {
                x0, y0, z0 := Coord2Loc(r1, i, j, k)
                x1, y2, z3 := Coord2Loc(r2, i, j, k)
                // Find if coordinates are valid in r1 and r2 independently
                r1_valid = r1x_valid && r1y_valid && r1z_valid
                r2_valid = r2x_valid && r2y_valid && r2z_valid
                
                // Choose the block based on validity and layer
                switch {
                    case r1_valid && r2_valid:  // If they are both valid, choose by layer
                        if r1.layer[i][j][k] > r2.layer[i][j][k] {
                            block, ang, layer = r1.bp.space[i][j][k], r1.bp.rotation[i][j][k], r1.layer[i][j][k]
                        } else {
                            block, ang, layer = r2.bp.space[i][j][k], r2.bp.rotation[i][j][k], r2.layer[i][j][k]
                        }
                    case r1_valid:
                        block, ang, layer = r1.bp.space[i][j][k], r1.bp.rotation[i][j][k], r1.layer[i][j][k]
                    case r2_valid:
                        block, ang, layer = r2.bp.space[i][j][k], r2.bp.rotation[i][j][k], r2.layer[i][j][k]
                    default:    // If none are valid, an empty block is passed
                        block, ang, layer = 0, 0, 0
                }

                // Set the block at out[coord]
                blockM[i][j][k] = block
                angM[i][j][k]   = ang
                layerM[i][j][k] = layer
            }
        }
    }
    
    // Return a region with space parameter as out, and the size as dx dy dz
    bp := Blueprint {
        name: name,
        space: *blockM,
        rotation: *angM,
        dx, dy, dz: dx, dy, dz}
    out := Region {
        bp: &bp,
        layer: *layerM,
        x, y, z: x0, y0, z0}
    return
}