from qewton.config import Variable


def plot_data(data, data_config):
    assert data fits data_config
    
    
    batch = SliderAxis(
        Variable("batch", 1),    
        init_state=0,
        minimum=0,
        maximum=19,
        step=1,
    )


    X = Variable("x", dim=1)
    Y = Variable("y", dim=1)

    config = PlotConfiguration(
        axes=[
            batch,
            XAxis(X),
            YAxis(Y),
        ]
    )

    plot = ImagePlot(
        images,
        config,
        title="Images",
    )

    figure = Figure(plot, title="Image Plot")

    app = DashApplication.create(figure)

    app.run(debug=True)#, jupyter_mode="inline")