from scheduler.scheduler import DiscreteDistribution, UniformDistribution, Time

def test_joint_distribution():
    d1 = UniformDistribution(Time(hours=12), Time(hours=13))
    d2 = UniformDistribution(Time(hours=10), Time(hours=11))
    d3 = d1.joint(d2)
    assert d3.density_at(Time(hours=12).minutes_since_midnight) > 0.099
    assert d3.density_at(Time(hours=10).minutes_since_midnight) > 0.099
    assert d3.density_at(Time(hours=1).minutes_since_midnight) < 0.002
