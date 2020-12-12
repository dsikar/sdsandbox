import numpy as np
from utils.steerlib import GetSteeringFromtcpflow, plotMultipleSteeringAngles

def plot1():
    """
    Intensity multiplier 1, 20201207091932_nvidia1.h5
    """

    # intensity multiplier = 1
    p = []
    # No rain mult 1 (shaw we have a "no rain" for every multiplier ?)
    sa = GetSteeringFromtcpflow('../../trained_models/nvidia1/tcpflow/20201207091932_nvidia1_no_rain_tcpflow.log')
    sarr = np.asarray(sa)
    pa = sarr[:,0]
    p.append([pa, 'no rain'])
    # light mult 1
    sa = GetSteeringFromtcpflow('../../trained_models/nvidia1/tcpflow/20201207091932_nvidia1_light_rain_mult_1_tcpflow.log')
    sarr = np.asarray(sa)
    pa = sarr[:,0]
    p.append([pa, 'light rain'])
    # heavy mult 1
    sa = GetSteeringFromtcpflow('../../trained_models/nvidia1/tcpflow/20201207091932_nvidia1_heavy_10_mult_1_tcpflow.log')
    sarr = np.asarray(sa)
    pa = sarr[:,0]
    p.append([pa, 'heavy rain slant +-10'])
    # torrential mult 1
    sa = GetSteeringFromtcpflow('../../trained_models/nvidia1/tcpflow/20201207091932_nvidia1_torrential_20_mult_1_tcpflow.log')
    sarr = np.asarray(sa)
    pa = sarr[:,0]
    p.append([pa, 'torrential rain slant +-20'])

    plotMultipleSteeringAngles(p, 25, True, "Generated Track intensity multiplier 1", "20201207091932_nvidia1.h5", 'tcpflow log predicted')

def plot2():
    """
    Intensity multiplier 4, 20201207091932_nvidia1.h5
    """
    # intensity multiplier = 4
    p = []
    # No rain mult 1 (shaw we have a "no rain" for every multiplier ?)
    sa = GetSteeringFromtcpflow('../../trained_models/nvidia1/tcpflow/20201207091932_nvidia1_no_rain_tcpflow.log')
    sarr = np.asarray(sa)
    pa = sarr[:, 0]
    p.append([pa, 'no rain'])
    # light mult 1
    sa = GetSteeringFromtcpflow('../../trained_models/nvidia1/tcpflow/20201207091932_nvidia1_light_rain_mult_4_tcpflow.log')
    sarr = np.asarray(sa)
    pa = sarr[:, 0]
    p.append([pa, 'light rain'])
    # heavy mult 1
    sa = GetSteeringFromtcpflow('../../trained_models/nvidia1/tcpflow/20201207091932_nvidia1_heavy_10_mult_4_tcpflow.log')
    sarr = np.asarray(sa)
    pa = sarr[:, 0]
    p.append([pa, 'heavy rain slant +-10'])
    # torrential mult 1
    sa = GetSteeringFromtcpflow('../../trained_models/nvidia1/tcpflow/20201207091932_nvidia1_torrential_20_mult_4_tcpflow.log')
    sarr = np.asarray(sa)
    pa = sarr[:, 0]
    p.append([pa, 'torrential rain slant +-20'])

    plotMultipleSteeringAngles(p, 25, True, "Generated Track intensity multiplier 4", "20201207091932_nvidia1.h5",
                               'tcpflow log predicted')

def plot3():
    """
    Intensity multiplier 8, 20201207091932_nvidia1.h5
    """

    # intensity multiplier = 8
    p = []
    # No rain mult 1 (shaw we have a "no rain" for every multiplier ?)
    sa = GetSteeringFromtcpflow('../../trained_models/nvidia1/tcpflow/20201207091932_nvidia1_no_rain_tcpflow.log')
    sarr = np.asarray(sa)
    pa = sarr[:, 0]
    p.append([pa, 'no rain'])
    # light mult 1
    sa = GetSteeringFromtcpflow(
        '../../trained_models/nvidia1/tcpflow/20201207091932_nvidia1_light_rain_mult_8_tcpflow.log')
    sarr = np.asarray(sa)
    pa = sarr[:, 0]
    p.append([pa, 'light rain'])
    # heavy mult 1
    sa = GetSteeringFromtcpflow(
        '../../trained_models/nvidia1/tcpflow/20201207091932_nvidia1_heavy_10_mult_8_tcpflow.log')
    sarr = np.asarray(sa)
    pa = sarr[:, 0]
    p.append([pa, 'heavy rain slant +-10'])
    # torrential mult 1
    sa = GetSteeringFromtcpflow(
        '../../trained_models/nvidia1/tcpflow/20201207091932_nvidia1_torrential_20_mult_8_tcpflow.log')
    sarr = np.asarray(sa)
    pa = sarr[:, 0]
    p.append([pa, 'torrential rain slant +-20'])

    plotMultipleSteeringAngles(p, 25, True, "Generated Track intensity multiplier 8", "20201207091932_nvidia1.h5",
                               'tcpflow log predicted')

def plot4():
    """
    Intensity multiplier 1, 20201207192948_nvidia2.h5
    """

    # intensity multiplier = 1
    p = []
    # No rain mult 1 (shaw we have a "no rain" for every multiplier ?)
    sa = GetSteeringFromtcpflow('../../trained_models/nvidia2/tcpflow/20201207192948_nvidia2_no_rain_tcpflow.log')
    sarr = np.asarray(sa)
    pa = sarr[:,0]
    p.append([pa, 'no rain'])
    # light mult 1
    sa = GetSteeringFromtcpflow('../../trained_models/nvidia2/tcpflow/20201207192948_nvidia2_light_rain_mult_1_tcpflow.log')
    sarr = np.asarray(sa)
    pa = sarr[:,0]
    p.append([pa, 'light rain'])
    # heavy mult 1
    sa = GetSteeringFromtcpflow('../../trained_models/nvidia2/tcpflow/20201207192948_nvidia2_heavy_10_mult_1_tcpflow.log')
    sarr = np.asarray(sa)
    pa = sarr[:,0]
    p.append([pa, 'heavy rain slant +-10'])
    # torrential mult 1
    sa = GetSteeringFromtcpflow('../../trained_models/nvidia2/tcpflow/20201207192948_nvidia2_torrential_20_mult_1_tcpflow.log')
    sarr = np.asarray(sa)
    pa = sarr[:,0]
    p.append([pa, 'torrential rain slant +-20'])

    plotMultipleSteeringAngles(p, 25, True, "Generated Track intensity multiplier 1", "20201207192948_nvidia2.h5", 'tcpflow log predicted')

def plot5():
    """
    Intensity multiplier 4, 20201207192948_nvidia2.h5
    """
    # intensity multiplier = 4
    p = []
    # No rain mult 1 (shaw we have a "no rain" for every multiplier ?)
    sa = GetSteeringFromtcpflow('../../trained_models/nvidia2/tcpflow/20201207192948_nvidia2_no_rain_tcpflow.log')
    sarr = np.asarray(sa)
    pa = sarr[:, 0]
    p.append([pa, 'no rain'])
    # light mult 1
    sa = GetSteeringFromtcpflow('../../trained_models/nvidia2/tcpflow/20201207192948_nvidia2_light_rain_mult_4_tcpflow.log')
    sarr = np.asarray(sa)
    pa = sarr[:, 0]
    p.append([pa, 'light rain'])
    # heavy mult 1
    sa = GetSteeringFromtcpflow('../../trained_models/nvidia2/tcpflow/20201207192948_nvidia2_heavy_10_mult_4_tcpflow.log')
    sarr = np.asarray(sa)
    pa = sarr[:, 0]
    p.append([pa, 'heavy rain slant +-10'])
    # torrential mult 1
    sa = GetSteeringFromtcpflow('../../trained_models/nvidia2/tcpflow/20201207192948_nvidia2_torrential_20_mult_4_tcpflow.log')
    sarr = np.asarray(sa)
    pa = sarr[:, 0]
    p.append([pa, 'torrential rain slant +-20'])

    plotMultipleSteeringAngles(p, 25, True, "Generated Track intensity multiplier 4", "20201207192948_nvidia2.h5",
                               'tcpflow log predicted')

def plot6():
    """
    Intensity multiplier 8, 20201207192948_nvidia2.h5
    """

    # intensity multiplier = 8
    p = []
    # No rain mult 1 (shaw we have a "no rain" for every multiplier ?)
    sa = GetSteeringFromtcpflow('../../trained_models/nvidia2/tcpflow/20201207192948_nvidia2_no_rain_tcpflow.log')
    sarr = np.asarray(sa)
    pa = sarr[:, 0]
    p.append([pa, 'no rain'])
    # light mult 1
    sa = GetSteeringFromtcpflow(
        '../../trained_models/nvidia2/tcpflow/20201207192948_nvidia2_light_rain_mult_8_tcpflow.log')
    sarr = np.asarray(sa)
    pa = sarr[:, 0]
    p.append([pa, 'light rain'])
    # heavy mult 1
    sa = GetSteeringFromtcpflow(
        '../../trained_models/nvidia2/tcpflow/20201207192948_nvidia2_heavy_10_mult_8_tcpflow.log')
    sarr = np.asarray(sa)
    pa = sarr[:, 0]
    p.append([pa, 'heavy rain slant +-10'])
    # torrential mult 1
    sa = GetSteeringFromtcpflow(
        '../../trained_models/nvidia2/tcpflow/20201207192948_nvidia2_torrential_20_mult_8_tcpflow.log')
    sarr = np.asarray(sa)
    pa = sarr[:, 0]
    p.append([pa, 'torrential rain slant +-20'])

    plotMultipleSteeringAngles(p, 25, True, "Generated Track intensity multiplier 8", "20201207192948_nvidia2.h5",
                               'tcpflow log predicted')
if __name__ == "__main__":
    # 20201207091932_nvidia1.h5
    # mult_1
    # plot1()
    # mult_4
    # plot2()
    # mult_8
    # plot3()
    # 20201207192948_nvidia2.h5
    # mult_1
    # plot4()
    # mult_4
    # plot5()
    # mult_8
    # plot6()