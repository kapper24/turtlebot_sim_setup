# the ray casting algorithm is inspiret by:
# https://theshoemaker.de/2016/02/ray-casting-in-2d-grids/
# https://github.com/pfirsich/Ray-casting-test/blob/master/main.lua
# which is inspired by “A Fast Voxel Traversal Algorithm for Ray Tracing” by John Amanatides and Andrew Woo

import torch

from sys import path
from os.path import dirname as dir
path.append(dir(path[0].replace("/robotexploration", "")))
__package__ = "probmind"

from scripts.misc import probabilistic_OR_independent
from barrierfunctions import sigmoidSS

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # 3.1415927410125732
relu = torch.nn.ReLU()


@torch.jit.script
def getHelpers(cellSize, pos, rayDir):
    tile = torch.floor(pos / cellSize).long() + 1

    if rayDir == 0:
        dTile = torch.tensor([0], dtype=torch.long)
        dt = torch.tensor([0])
        ddt = torch.tensor([0])
    elif rayDir > 0:
        dTile = torch.tensor([1], dtype=torch.long)
        dt = ((tile + torch.tensor([0.])) * cellSize - pos) / rayDir
        ddt = dTile * cellSize / rayDir
    else:
        dTile = torch.tensor([-1], dtype=torch.long)
        dt = ((tile - torch.tensor([1.])) * cellSize - pos) / rayDir
        ddt = dTile * cellSize / rayDir

    tile = tile - torch.tensor([1], dtype=torch.long)

    return tile, dTile, dt, ddt


# @torch.jit.script
def castRayVectorDirClosest(grid, rayStart, rayDir, maxdist, flipped_y_axis=True):
    grid_shape = grid.size()
    grid_width = grid_shape[1]
    grid_height = grid_shape[0]

    if flipped_y_axis:
        rayStartX = rayStart[0]
        rayStartY = grid_height - rayStart[1]
        rayDirX = rayDir[0]
        rayDirY = -rayDir[1]
    else:
        rayStartX = rayStart[0]
        rayStartY = rayStart[1]
        rayDirX = rayDir[0]
        rayDirY = rayDir[1]

    cellSize = torch.tensor([1.], dtype=torch.float)
    tileX, dtileX, dtX, ddtX = getHelpers(cellSize, rayStartX, rayDirX)
    tileY, dtileY, dtY, ddtY = getHelpers(cellSize, rayStartY, rayDirY)
    t = torch.tensor([0.], dtype=torch.float)
    t = torch.tensor([0.], dtype=torch.float)

    if dtX == 0:
        while tileX > 0 and tileX <= grid_width and tileY > 0 and tileY <= grid_height:
            if grid[tileY, tileX] > torch.tensor([0.]) or t > maxdist:
                if t > maxdist:
                    t = maxdist
                break
            tileY = tileY + dtileY
            dt = dtY
            t = t + dt
            dtY = dtY + ddtY - dt
    elif dtY == 0:
        while tileX > 0 and tileX <= grid_width and tileY > 0 and tileY <= grid_height:
            if grid[tileY, tileX] > torch.tensor([0.]) or t > maxdist:
                if t > maxdist:
                    t = maxdist
                break
            tileX = tileX + dtileX
            dt = dtX
            t = t + dt
            dtX = dtX + ddtX - dt
    else:
        while tileX > 0 and tileX <= grid_width and tileY > 0 and tileY <= grid_height:
            if grid[tileY, tileX] > torch.tensor([0.]) or t > maxdist:
                if t > maxdist:
                    t = maxdist
                break

            if (dtX != 0 and dtX < dtY) or dtY == 0:
                tileX = tileX + dtileX
                dt = dtX
                t = t + dt
                dtX = dtX + ddtX - dt
                dtY = dtY - dt
            else:
                tileY = tileY + dtileY
                dt = dtY
                t = t + dt
                dtX = dtX - dt
                dtY = dtY + ddtY - dt

    return t, tileY, tileX


# @torch.jit.script
def castRayAngleDirClosest(grid, rayStart, angle, maxdist):
    rayDir = torch.stack((torch.cos(angle), torch.sin(angle)), 0)
    t, tileY, tileX = castRayVectorDirClosest(grid, rayStart, rayDir, maxdist)

    return t, tileY, tileX, rayDir


# @torch.jit.script
def lidar(grid, center, N_beam_samples, maxdist):
    # angles = torch.arange(start=0, end=2 * torch.pi, step=torch.pi * 2 / N_beam_samples)
    angles = 2 * torch.pi * torch.rand(N_beam_samples)  # IMPORTANT TO CHANGE WHEN COMPARING over multiple runs!!!!

    t = torch.zeros(N_beam_samples, dtype=torch.float)
    tileY = torch.zeros(N_beam_samples, dtype=torch.long)
    tileX = torch.zeros(N_beam_samples, dtype=torch.long)
    points = torch.zeros(N_beam_samples, 2, dtype=torch.float)
    j = 0  # counter of point below maxDist
    for i in torch.arange(N_beam_samples):
        t_tmp, tileY_tmp, tileX_tmp, rayDir_tmp = castRayAngleDirClosest(grid, center, angles[i], maxdist)
        t[i] = t_tmp.item()
        tileY[i] = tileY_tmp.item()
        tileX[i] = tileX_tmp.item()
        if t[i] < maxdist:
            points[j, :] = center + t[i] * rayDir_tmp
            j = j + 1

    if j == 0:  # there was no points below max dist
        return t, tileY, tileX, None
    else:
        return t, tileY, tileX, points[torch.arange(j - 1), :]


def constraint(z_t, m, params):
    dist = torch.dist(m, z_t, p=2)

    # return barrier.expC(dist, scale, x_min)
    # return barrier.expCC(dist, scale, x_min, x_max)
    # return barrier.ReLU1F(dist, x_min, x_max)
    return sigmoidSS(dist, params["P_z_C1_scale"], params["x_min"], params["x_max"])


def simple2DrayTracing(z_s, params):
    maxdist = params["lidarParams"]["z_max"] * params["lidarParams"]["meter2pixel"]  # range in pixels
    N_beam_samples = params["lidarParams"]["N_lidar_beams_samples"]

    map_ = params["map"]
    map_shape = map_.size()

    rayStart = z_s * params["lidarParams"]["meter2pixel"]

    if (rayStart[0] > 0) and (rayStart[0] < map_shape[1] - 1) and (rayStart[1] > 0) and (rayStart[1] < map_shape[0] - 1):
        with torch.no_grad():
            t, tileY, tileX, points = lidar(map_, rayStart, N_beam_samples, maxdist)
        if points is not None:
            points[:, 0] = points[:, 0] / params["lidarParams"]["meter2pixel"]
            points[:, 1] = points[:, 1] / params["lidarParams"]["meter2pixel"]

            tmp = torch.tensor(0.)
            for i in range(len(points)):
                if map_[tileY[i], tileX[i]] > 0.8:  # only consider if there is a high probability of the cell being occupied
                    c_tmp = constraint(z_s, points[i, :], params)
                    tmp = probabilistic_OR_independent([tmp, c_tmp])
            c_ = tmp
        else:  # no objects near
            c_ = torch.tensor([0.], dtype=torch.float)
    else:
        c_ = torch.tensor([1.], dtype=torch.float)

    return c_


def P_z_C1(z_s, params):
    p_z_C_ = torch.tensor([1.], dtype=torch.float) - simple2DrayTracing(z_s, params)
    return p_z_C_

























# @torch.jit.script
def castRayVectorDirALL(grid, rayStart, rayDir, maxdist, flipped_y_axis=True):
    grid_shape = grid.size()
    grid_width = grid_shape[1]
    grid_height = grid_shape[0]

    if flipped_y_axis:
        rayStartX = rayStart[0]
        rayStartY = grid_height - rayStart[1]
        rayDirX = rayDir[0]
        rayDirY = -rayDir[1]
    else:
        rayStartX = rayStart[0]
        rayStartY = rayStart[1]
        rayDirX = rayDir[0]
        rayDirY = rayDir[1]

    cellSize = torch.tensor([1.], dtype=torch.float)
    tileX, dtileX, dtX, ddtX = getHelpers(cellSize, rayStartX, rayDirX)
    tileY, dtileY, dtY, ddtY = getHelpers(cellSize, rayStartY, rayDirY)
    t = torch.tensor([0.], dtype=torch.float)
    t = torch.tensor([0.], dtype=torch.float)

    t_out = []
    tileY_out = []
    tileX_out = []
    gridValues = []
    if dtX == 0:
        while tileX >= 0 and tileX <= grid_width - 1 and tileY >= 0 and tileY <= grid_height - 1:
            if t > maxdist:
                t = maxdist
                if grid[tileY, tileX] > torch.tensor([0.]):
                    t_out.append(t)
                    tileY_out.append(tileY)
                    tileX_out.append(tileX)
                    gridValues.append(grid[tileY, tileX].item())
                break

            if grid[tileY, tileX] > torch.tensor([0.]):
                t_out.append(t)
                tileY_out.append(tileY)
                tileX_out.append(tileX)
                gridValues.append(grid[tileY, tileX].item())
            tileY = tileY + dtileY
            dt = dtY
            t = t + dt
            dtY = dtY + ddtY - dt
    elif dtY == 0:
        while tileX >= 0 and tileX <= grid_width - 1 and tileY >= 0 and tileY <= grid_height - 1:
            if t > maxdist:
                t = maxdist
                if grid[tileY, tileX] > torch.tensor([0.]):
                    t_out.append(t)
                    tileY_out.append(tileY)
                    tileX_out.append(tileX)
                    gridValues.append(grid[tileY, tileX].item())
                break

            if grid[tileY, tileX] > torch.tensor([0.]):
                t_out.append(t)
                tileY_out.append(tileY)
                tileX_out.append(tileX)
                gridValues.append(grid[tileY, tileX].item())

            tileX = tileX + dtileX
            dt = dtX
            t = t + dt
            dtX = dtX + ddtX - dt
    else:
        while tileX >= 0 and tileX <= grid_width - 1 and tileY >= 0 and tileY <= grid_height - 1:
            if t > maxdist:
                t = maxdist
                if grid[tileY, tileX] > torch.tensor([0.]):
                    t_out.append(t)
                    tileY_out.append(tileY)
                    tileX_out.append(tileX)
                    gridValues.append(grid[tileY, tileX].item())
                break

            if grid[tileY, tileX] > torch.tensor([0.]):
                t_out.append(t)
                tileY_out.append(tileY)
                tileX_out.append(tileX)
                gridValues.append(grid[tileY, tileX].item())

            if dtX < dtY:
                tileX = tileX + dtileX
                dt = dtX
                t = t + dt
                dtX = dtX + ddtX - dt
                dtY = dtY - dt
            else:
                tileY = tileY + dtileY
                dt = dtY
                t = t + dt
                dtX = dtX - dt
                dtY = dtY + ddtY - dt

    t_out = torch.FloatTensor(t_out)
    tileY_out = torch.FloatTensor(tileY_out).long()
    tileX_out = torch.FloatTensor(tileX_out).long()
    gridValues = torch.FloatTensor(gridValues)
    return t_out, tileY_out, tileX_out, gridValues


def castRayAngleDirALL(grid, rayStart, angle, maxdist):
    rayDir = torch.stack((torch.cos(angle), torch.sin(angle)), 0)
    t, tileY, tileX, gridValues = castRayVectorDirALL(grid, rayStart, rayDir, maxdist)

    return t, tileY, tileX, gridValues


def collisions(grid, startPoint, endPoint):
    rayDir = endPoint - startPoint
    maxdist = torch.linalg.norm(rayDir)
    rayDir = rayDir / maxdist
    t, tileY, tileX, gridValues = castRayVectorDirALL(grid, startPoint, rayDir, maxdist)
    if t.numel() > 0:
        firstPoint = startPoint + rayDir * t[0]
    else:
        firstPoint = torch.FloatTensor([])
    return t, tileY, tileX, gridValues, firstPoint


def P_z_C2(z_s, z_s_m1, params):
    map_ = params["map"]
    map_shape = map_.size()
    startPoint = z_s * params["lidarParams"]["meter2pixel"]
    endPoint = z_s_m1 * params["lidarParams"]["meter2pixel"]

    if (startPoint[0] > 0) and (startPoint[0] < map_shape[1] - 1) and (startPoint[1] > 0) and (startPoint[1] < map_shape[0] - 1) and (endPoint[0] > 0) and (endPoint[0] < map_shape[1] - 1) and (endPoint[1] > 0) and (endPoint[1] < map_shape[0] - 1):
        with torch.no_grad():
            t, tileY, tileX, gridValues, firstPoint = collisions(map_, startPoint, endPoint)
        if firstPoint.numel() > 0:
            firstPoint[0] = firstPoint[0] / params["lidarParams"]["meter2pixel"]
            firstPoint[1] = firstPoint[1] / params["lidarParams"]["meter2pixel"]

            dist = torch.dist(z_s, firstPoint, p=2)
            c_ = torch.exp(-dist * params["P_z_C2_scale"])
        else:
            c_ = torch.tensor([1.], dtype=torch.float)
    else:
        c_ = torch.tensor([0.], dtype=torch.float)

    p_z_C_ = c_
    # p_z_C_ = torch.tensor([1.],dtype=torch.float) - c_
    return p_z_C_
