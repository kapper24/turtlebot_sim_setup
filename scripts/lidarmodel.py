   # the ray casting algorithm is inspiret by:
# https://theshoemaker.de/2016/02/ray-casting-in-2d-grids/
# https://github.com/pfirsich/Ray-casting-test/blob/master/main.lua
# which is inspired by “A Fast Voxel Traversal Algorithm for Ray Tracing” by John Amanatides and Andrew Woo

import torch
import pyro
from pyro.contrib.autoname import scope
import pyro.distributions as dist
from pyro.distributions.torch_distribution import TorchDistributionMixin

from Trunc import TruncatedNormal


torch.pi = torch.acos(torch.zeros(1)).item() * 2  # 3.1415927410125732


# class TruncNormal(TruncatedNormal.TruncatedNormal, TorchDistributionMixin):
class TruncNormal(TruncatedNormal, TorchDistributionMixin):
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(TruncNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.a = self.a.expand(batch_shape)
        new.b = self.b.expand(batch_shape)
        super(TruncNormal, new).__init__(new.loc, new.scale, new.a, new.b)
        new._validate_args = self._validate_args
        return new


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

    t_out = torch.tensor([], dtype=torch.float)

    tileY_out = []
    tileX_out = []
    gridValues = []
    if dtX == 0:
        while tileX >= 0 and tileX <= grid_width - 1 and tileY >= 0 and tileY <= grid_height - 1:
            if t > maxdist:
                t = maxdist
                if grid[tileY, tileX] > torch.tensor([0.]):
                    t_out = torch.cat((t_out, t), dim=0)
                    tileY_out.append(tileY)
                    tileX_out.append(tileX)
                    gridValues.append(grid[tileY, tileX].item())
                break

            if grid[tileY, tileX] > torch.tensor([0.]):
                tileY_out.append(tileY)
                tileX_out.append(tileX)
                gridValues.append(grid[tileY, tileX].item())
                t_out = torch.cat((t_out, t), dim=0)  # <-- this works with pyro
                # but it does yield a wrong gradient
                # As a quick fix we use the euclidean distance instead
                # point1 = torch.vstack([rayStartX, rayStartY])
                # point2 = t * torch.vstack([rayDirX, rayDirY])
                # v = point2 - point1
                # # v = torch.vstack([tileX, tileY]) - torch.vstack([rayStartX, rayStartY])
                # dist = torch.linalg.norm(v)
                # print("dist: " + str(dist) + "  t: " + str(t))
                # t_out = torch.cat((t_out, dist.view(1)), dim=0)

            tileY = tileY + dtileY
            dt = dtY
            t = t + dt
            dtY = dtY + ddtY - dt
    elif dtY == 0:
        while tileX >= 0 and tileX <= grid_width - 1 and tileY >= 0 and tileY <= grid_height - 1:
            if t > maxdist:
                t = maxdist
                if grid[tileY, tileX] > torch.tensor([0.]):
                    t_out = torch.cat((t_out, t), dim=0)
                    tileY_out.append(tileY)
                    tileX_out.append(tileX)
                    gridValues.append(grid[tileY, tileX].item())
                break

            if grid[tileY, tileX] > torch.tensor([0.]):
                tileY_out.append(tileY)
                tileX_out.append(tileX)
                gridValues.append(grid[tileY, tileX].item())
                t_out = torch.cat((t_out, t), dim=0)  # <-- this works with pyro
                # but it does yield a wrong gradient
                # As a quick fix we use the euclidean distance instead
                # point1 = torch.vstack([rayStartX, rayStartY])
                # point2 = t * torch.vstack([rayDirX, rayDirY])
                # v = point2 - point1
                # # v = torch.vstack([tileX, tileY]) - torch.vstack([rayStartX, rayStartY])
                # dist = torch.linalg.norm(v)
                # print("dist: " + str(dist) + "  t: " + str(t))
                # t_out = torch.cat((t_out, dist.view(1)), dim=0)

            tileX = tileX + dtileX
            dt = dtX
            t = t + dt
            dtX = dtX + ddtX - dt
    else:
        while tileX >= 0 and tileX <= grid_width - 1 and tileY >= 0 and tileY <= grid_height - 1:
            if t > maxdist:
                t = maxdist
                if grid[tileY, tileX] > torch.tensor([0.]):
                    tileY_out.append(tileY)
                    tileX_out.append(tileX)
                    gridValues.append(grid[tileY, tileX].item())
                    t_out = torch.cat((t_out, t), dim=0)
                break

            if grid[tileY, tileX] > torch.tensor([0.]):
                tileY_out.append(tileY)
                tileX_out.append(tileX)
                gridValues.append(grid[tileY, tileX].item())
                t_out = torch.cat((t_out, t), dim=0)  # <-- this works with pyro
                # but it does yield a wrong gradient
                # As a quick fix we use the euclidean distance instead
                # point1 = torch.vstack([rayStartX, rayStartY])
                # point2 = t * torch.vstack([rayDirX, rayDirY])
                # v = point2 - point1
                # # v = torch.vstack([tileX, tileY]) - torch.vstack([rayStartX, rayStartY])
                # dist = torch.linalg.norm(v)
                # print("dist: " + str(dist) + "  t: " + str(t))
                # t_out = torch.cat((t_out, dist.view(1)), dim=0)

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

    tileY_out = torch.FloatTensor(tileY_out).long()
    tileX_out = torch.FloatTensor(tileX_out).long()
    gridValues = torch.FloatTensor(gridValues)
    return t_out, tileY_out, tileX_out, gridValues


def castRayAngleDirALL(grid, rayStart, angle, maxdist):
    rayDir = torch.stack((torch.cos(angle), torch.sin(angle)), 0)
    t, tileY, tileX, gridValues = castRayVectorDirALL(grid, rayStart, rayDir, maxdist)

    return t, tileY, tileX, gridValues


def p_z_Map_beam_sampler(position, angle, map_grid_probabilities, lidarParams):
    # Samples from the prior distribution over LTM
    # Notice that the the state sample, z_s, and sample from PB, z_PB, should not alter the sampling probability
    # of LTM, but should only be used to avoid sampling parts of the LTM that is independent of the current
    # situation for computational efficiency!
    # position: current state sample
    # z_PB: current sample from PB - in this case a tuble of an ray angle and sampled distance
    # LTM: variables storing current content of the LTM
    # Returns
    # z_LTM: a sample of the relevant LTM

    rayStart = position * lidarParams["meter2pixel"]  # torch.stack((P_z_s_0_map_idx_x, P_z_s_0_map_idx_y))

    t, tileY, tileX, gridValues = castRayAngleDirALL(map_grid_probabilities, rayStart, angle, lidarParams["z_max"] * lidarParams["meter2pixel"])

    z_n_star = None
    if len(gridValues) != 0:
        for i in range(len(gridValues)):
            z_LTM_X_Y = pyro.sample('z_LTM_Map_{}_{}'.format(tileX[i], tileY[i]), dist.Bernoulli(gridValues[i]))
            if z_LTM_X_Y:
                z_n_star = t[i] / lidarParams["meter2pixel"]
                break

    return z_n_star


class lidarSubsampler():
    N_beams = None
    plate_context = None
    indexes = None
    counter = 0

    def __init__(self):
        pass

    def generate_plate_context(suffix="", Count=False):
        lidarSubsampler.plate_context = lidarSubsampler.indexes
        # lidarSubsampler.plate_context = pyro.plate("map_lidar_beams_{}_".format(lidarSubsampler.counter)+suffix, lidarSubsampler.N_beams, subsample=lidarSubsampler.indexes)
        if Count:
            lidarSubsampler.counter = lidarSubsampler.counter + 1

    def sample_all():
        lidarSubsampler.indexes = range(lidarSubsampler.N_beams)

    def sample_specific(indexes):
        # indexes: indexes to use
        lidarSubsampler.indexes = indexes

    def sample_random(N_samples):
        # N_samples: number of samples
        probs = torch.ones(lidarSubsampler.N_beams) / lidarSubsampler.N_beams
        lidarSubsampler.indexes = torch.multinomial(probs, N_samples).tolist()


def lidar_generate_obs_labels(lidarParams):
    lidarSubsampler.N_beams = lidarParams["N_lidar_beams"]

    if isinstance(lidarParams["N_lidar_beams_samples"], int):
        lidarSubsampler.sample_random(lidarParams["N_lidar_beams_samples"])
    elif type(lidarParams["N_lidar_beams_samples"]) == list:
        lidarSubsampler.sample_specific(lidarParams["N_lidar_beams_samples"])
    else:  # assume that all should be sampled
        lidarSubsampler.sample_all()

    lidarSubsampler.generate_plate_context()

    observation_labels = []
    for n in lidarSubsampler.indexes:
        observation_labels.append(str(n) + "/z_beam")
    return observation_labels


def p_z_Map_prior(position, map_grid_probabilities, lidarParams):
    # we could sample all the cells in the current belief of the map,
    # however that would be very inefficient. Therefore, we instead
    # only sample cells intersecting with the directions of the lidars
    # laser beams, and save the distance coresponding to the first cell
    # that is sampled to be occupied in those directions.
    # arg:

    z_Map = {}  # dict to contain the LTM samples related to the map

    for n in lidarSubsampler.plate_context:
        with scope(prefix=str(n)):
            angle = torch.tensor(n * (torch.pi * 2 / lidarSubsampler.N_beams))
            z_beam_n_star = p_z_Map_beam_sampler(position, angle, map_grid_probabilities, lidarParams)
            z_Map["z_beam_" + str(n) + "_star"] = z_beam_n_star

    return z_Map


def p_z_beam_prior(lidarParams):
    # the following assumes that there always is an object in the range [0,z_max]!
    # consider if there should be added some likelyhood to not sense an object within this range,
    # e.g. by adding a z_max part as in "signel_beam_forward"

    beam_max_range = lidarParams["z_max"]

    # P_hit = 0.99999
    # P_rand = 0.00001
    # P_max = 0.0
    # P_short = 0.0 # not implemented yet! => should always be zero

    # assignment_probs = torch.tensor([P_hit, P_rand])
    # z_beam_category = pyro.sample('z_beam_category', dist.Categorical(assignment_probs), infer={"enumerate": "sequential"})

    # The uniform distribution in pytorch is on the interval [a,b), thus excluding b
    # leading to an incosistency of likelyhood at z_k=z_max if b=z_max, since using the uniform distribution
    # on the interval [0,z_max) would be the same as saying that the likelyhood of the sensor generating a
    # random measurement at z_k=z_max is zero.
    # to make sure that z_max is included in the interval, flip the distribution!
    p_z_beam = dist.Uniform(0.0, beam_max_range)
    z_beam = pyro.sample('z_beam', p_z_beam)

    return z_beam


def p_z_beam_posterior(z_beam_star, lidarParams):
    # made according to the book "Probabilistic Robotics"
    # z_k: value to evaluate the likelyhood of
    # z_beam_star: the "true" distance found be raytracing in the (sampled) map
    # beam_max_range: the max measurement distance of the beam sensor

    # Model params:
    beam_max_range = lidarParams["z_max"]
    sigma_hit = lidarParams["sigma_hit"]
    lambda_short = lidarParams["lambda_short"]
    P_hit = lidarParams["P_hit"]
    P_rand = lidarParams["P_rand"]
    P_max = lidarParams["P_max"]
    P_short = lidarParams["P_short"]  # not implemented yet! => should always be zero

    if z_beam_star is None:
        z_beam_star_ = beam_max_range
        P_rand = P_rand + P_hit
        # P_max = P_max + P_hit
        P_hit = 0.0
    else:
        z_beam_star_ = z_beam_star  # internal variable

    assignment_probs = torch.tensor([P_hit, P_rand, P_max, P_short])
    z_beam_category = pyro.sample('z_beam_category', dist.Categorical(assignment_probs), infer={"enumerate": "sequential"})

    if z_beam_category == 0:  # hit
        p_z_beam = TruncNormal(z_beam_star_, sigma_hit, 0.0, beam_max_range)

    elif z_beam_category == 1:  # random
        p_z_beam = dist.Uniform(0.0, beam_max_range)

    elif z_beam_category == 2:  # max
        p_z_beam = dist.Delta(beam_max_range)

    else:  # z_beam_category == 1: # short - TruncExp not implemented yet
        raise ValueError('TruncExp not implemented')
        # p_z_beam = TruncExp(rate=lambda_short, 0, beam_max_range)

    z_beam = pyro.sample('z_beam', p_z_beam)

    return z_beam


def p_z_Lidar_prior(lidarParams):

    z_lidar = {}  # dict to contain the PB samples related to the lidar

    for n in lidarSubsampler.plate_context:
        with scope(prefix=str(n)):
            z_beam_n = p_z_beam_prior(lidarParams)
            z_lidar["z_beam_" + str(n)] = z_beam_n

    return z_lidar


def p_z_Lidar_posterior(z_Map, lidarParams):
    z_lidar = {}  # dict to contain the PB samples related to the lidar

    for n in lidarSubsampler.plate_context:
        with scope(prefix=str(n)):
            z_beam_n_star = z_Map["z_beam_" + str(n) + "_star"]
            z_beam_n = p_z_beam_posterior(z_beam_n_star, lidarParams)
            z_lidar["z_beam_" + str(n)] = z_beam_n

    return z_lidar

