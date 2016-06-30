package world3D

func Cube(blk Block, dx, dy, dz uint) Blueprint {
    var space [dx][dy][dz]Block
    for x := range dx {
        for y := range dy {
            for z := range dz {
                space[x][y][z] = blk
            }
        }
    }
    return NewBlueprint(space[:][:][:])
}