import torch
import torch.nn as nn

class KLDloss(nn.Module):
    def __init__(self, taf=1.0, reduction="none"):
        super(KLDloss, self).__init__()
        self.reduction = reduction
        self.taf = taf

    def forward(self, pred, target): # pred [[x,y,w,h,angle], ...]
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 5)
        target = target.view(-1, 5)

        delta_x = pred[:, 0] - target[:, 0]
        delta_y = pred[:, 1] - target[:, 1]
        pre_angle_radian = 3.141592653589793 * pred[:, 4] / 180.0
        targrt_angle_radian = 3.141592653589793 * target[:, 4] / 180.0
        delta_angle_radian = pre_angle_radian - targrt_angle_radian

        kld =  0.5 * (
                        4 * torch.pow( ( delta_x.mul(torch.cos(targrt_angle_radian)) + delta_y.mul(torch.sin(targrt_angle_radian)) ), 2) / torch.pow(target[:, 2], 2)
                      + 4 * torch.pow( ( delta_y.mul(torch.cos(targrt_angle_radian)) - delta_x.mul(torch.sin(targrt_angle_radian)) ), 2) / torch.pow(target[:, 3], 2)
                     )\
             + 0.5 * (
                        torch.pow(pred[:, 3], 2) / torch.pow(target[:, 2], 2) * torch.pow(torch.sin(delta_angle_radian), 2)
                      + torch.pow(pred[:, 2], 2) / torch.pow(target[:, 3], 2) * torch.pow(torch.sin(delta_angle_radian), 2)
                      + torch.pow(pred[:, 3], 2) / torch.pow(target[:, 3], 2) * torch.pow(torch.cos(delta_angle_radian), 2)
                      + torch.pow(pred[:, 2], 2) / torch.pow(target[:, 2], 2) * torch.pow(torch.cos(delta_angle_radian), 2)
                     )\
             + 0.5 * (
                        torch.log(torch.pow(target[:, 3], 2) / torch.pow(pred[:, 3], 2))
                      + torch.log(torch.pow(target[:, 2], 2) / torch.pow(pred[:, 2], 2))
                     )\
             - 1.0

        kld_loss = 1 - 1 / (self.taf + torch.log(kld + 1))

        # if self.reduction == "mean":
        #     kld_loss = loss.mean()
        # elif self.reduction == "sum":
        #     kld_loss = loss.sum()

        return kld_loss


def compute_kld_loss(targets, preds):
    with torch.no_grad():
        kld_loss_ts_ps = torch.zeros(0, preds.shape[0], device=targets.device)
        for target in targets:
            target = target.unsqueeze(0).repeat(preds.shape[0], 1)
            kld_loss_t_p = kld_loss(preds, target)
            kld_loss_ts_ps = torch.cat((kld_loss_ts_ps, kld_loss_t_p.unsqueeze(0)), dim=0)
    return kld_loss_ts_ps


def kld_loss(pred, target, taf=1.0):  # pred [[x,y,w,h,angle], ...]
    assert pred.shape[0] == target.shape[0]

    pred = pred.view(-1, 5)
    target = target.view(-1, 5)

    delta_x = pred[:, 0] - target[:, 0]
    delta_y = pred[:, 1] - target[:, 1]
    pre_angle_radian = 3.141592653589793 * pred[:, 4] / 180.0
    targrt_angle_radian = 3.141592653589793 * target[:, 4] / 180.0
    delta_angle_radian = pre_angle_radian - targrt_angle_radian

    kld = 0.5 * (
            4 * torch.pow((delta_x.mul(torch.cos(targrt_angle_radian)) + delta_y.mul(torch.sin(targrt_angle_radian))),
                          2) / torch.pow(target[:, 2], 2)
            + 4 * torch.pow((delta_y.mul(torch.cos(targrt_angle_radian)) - delta_x.mul(torch.sin(targrt_angle_radian))),
                            2) / torch.pow(target[:, 3], 2)
    ) \
          + 0.5 * (
                  torch.pow(pred[:, 3], 2) / torch.pow(target[:, 2], 2) * torch.pow(torch.sin(delta_angle_radian), 2)
                  + torch.pow(pred[:, 2], 2) / torch.pow(target[:, 3], 2) * torch.pow(torch.sin(delta_angle_radian), 2)
                  + torch.pow(pred[:, 3], 2) / torch.pow(target[:, 3], 2) * torch.pow(torch.cos(delta_angle_radian), 2)
                  + torch.pow(pred[:, 2], 2) / torch.pow(target[:, 2], 2) * torch.pow(torch.cos(delta_angle_radian), 2)
          ) \
          + 0.5 * (
                  torch.log(torch.pow(target[:, 3], 2) / torch.pow(pred[:, 3], 2))
                  + torch.log(torch.pow(target[:, 2], 2) / torch.pow(pred[:, 2], 2))
          ) \
          - 1.0

    kld_loss = 1 - 1 / (taf + torch.log(kld + 1))

    return kld_loss

# loss = KLDloss()
# pred = torch.tensor([[20, 20, 10, 10, -90], [20, 20, 20, 10, 90], [1, 0.5, 2, 1, 0]], dtype=torch.float32)
# target = torch.tensor([[20, 20, 10, 10, -90], [20, 20, 20, 10, 0], [0.5, 1, 2, 1, -90]], dtype=torch.float32)
# kld = kld_loss(pred, target)
# print(kld)


# pred = torch.tensor([[20, 20, 10, 10, -90], [20, 20, 20, 10, 90], [1, 0.5, 2, 1, 0]], dtype=torch.float32)
# target = torch.tensor([[20, 20, 10, 10, -90], [20, 20, 20, 10, 0]], dtype=torch.float32)
# kld = compute_kld_loss(target, pred)
# print(kld)
#
# print(torch.floor(torch.tensor(-9.9)))
def postprocess(distance, fun='log1p', tau=1.0):
    if fun == 'log1p':
        distance = torch.log1p(distance)
    elif fun == 'none':
        pass
    else:
        raise ValueError(f'Invalid non-linear function {fun}')

    if tau >= 1.0:
        return 1 - 1 / (tau + distance)
    else:
        return distance

def xy_stddev_pearson_2_xy_sigma(xy_stddev_pearson):
    _shape = xy_stddev_pearson.shape
    assert _shape[-1] == 5
    xy = xy_stddev_pearson[..., :2]
    stddev = xy_stddev_pearson[..., 2:4]
    pearson = xy_stddev_pearson[..., 4].clamp(min=1e-7 - 1, max=1 - 1e-7)
    covar = pearson * stddev.prod(dim=-1)
    var = stddev.square()
    sigma = torch.stack((var[..., 0],
                         covar,
                         covar,
                         var[..., 1]), dim=-1).reshape(_shape[:-1] + (2, 2))
    return xy, sigma

def kld_loss_(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True):
    # todo

    xy_p, Sigma_p = xy_stddev_pearson_2_xy_sigma(pred)
    xy_t, Sigma_t = xy_stddev_pearson_2_xy_sigma(target)

    _shape = xy_p.shape

    xy_p = xy_p.reshape(-1, 2)
    xy_t = xy_t.reshape(-1, 2)
    Sigma_p = Sigma_p.reshape(-1, 2, 2)
    Sigma_t = Sigma_t.reshape(-1, 2, 2)

    Sigma_p_inv = torch.stack((Sigma_p[..., 1, 1], -Sigma_p[..., 0, 1],
                               -Sigma_p[..., 1, 0], Sigma_p[..., 0, 0]),
                              dim=-1).reshape(-1, 2, 2)
    Sigma_p_inv = Sigma_p_inv / Sigma_p.det().unsqueeze(-1).unsqueeze(-1)

    dxy = (xy_p - xy_t).unsqueeze(-1)
    xy_distance = 0.5 * dxy.permute(0, 2, 1).bmm(Sigma_p_inv).bmm(
        dxy).view(-1)

    whr_distance = 0.5 * Sigma_p_inv.bmm(
        Sigma_t).diagonal(dim1=-2, dim2=-1).sum(dim=-1)

    Sigma_p_det_log = Sigma_p.det().log()
    Sigma_t_det_log = Sigma_t.det().log()
    whr_distance = whr_distance + 0.5 * (Sigma_p_det_log - Sigma_t_det_log)
    whr_distance = whr_distance - 1
    distance = (xy_distance / (alpha * alpha) + whr_distance)
    if sqrt:
        distance = distance.clamp(0).sqrt()

    distance = distance.reshape(_shape[:-1])

    return postprocess(distance, fun=fun, tau=tau)

# #loss = KLDloss()
# pred = torch.tensor([[20, 20, 10, 10, 10], [20, 20, 20, 10, 10], [1, 0.5, 2, 1, 0]], dtype=torch.float32)
# target = torch.tensor([[20, 20, 10, 10,5], [20, 20, 20, 10,5], [1.2, 1, 2, 1, 0]], dtype=torch.float32)
# kld = kld_loss(pred, target)
# print(kld)
# kld = kld_loss_(pred, target)
# print(kld)
# a = torch.tensor([[0.20],[0.40],[0.50]]).sigmoid()
# print(a)
# print(a.shape)
# print(torch.floor(a*180))
# pred = torch.tensor([[20, 20, 10, 10, -90], [20, 20, 20, 10, 90], [1, 0.5, 2, 1, 0]], dtype=torch.float32)
# target = torch.tensor([[20, 20, 10, 10, -90], [20, 20, 20, 10, 0]], dtype=torch.float32)
# kld = compute_kld_loss(target, pred)
# print(kld)
#
# print(torch.floor(torch.tensor(-9.9)))