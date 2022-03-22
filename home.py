from flask import Flask, render_template, request, redirect, url_for
app = Flask(__name__)
from metrics import prepare_data_stock, prepare_data_benchmark, prepare_data_ALL, get_CAGR_stock, get_CAGR_benchmark, get_volatility, get_sharpe_ratio, get_correlation, get_alpha, get_sortino_ratio, get_maximum_drawdown, get_calmar_ratio, get_VaR_90, get_VaR_95, get_VaR_99


@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        ticker_id = request.form["ticker_id"]
        return redirect(url_for("ticker", ticker_id=ticker_id))
    else:
        return render_template('home.html')


@app.route("/ticker/<ticker_id>")
def ticker(ticker_id):
    CAGR_stock = get_CAGR_stock()
    CAGR_benchmark = get_CAGR_benchmark()
    volatility = get_volatility()
    sharpe = get_sharpe_ratio()
    correlation = get_correlation()
    alpha = get_alpha()
    sortino = get_sortino_ratio()
    maximum_drawdown = get_maximum_drawdown()
    calmar = get_calmar_ratio()
    VaR_90 = get_VaR_90()
    VaR_95 = get_VaR_95()
    VaR_99 = get_VaR_99()
    return render_template('home2.html', usr=ticker_id,
                           CAGR_stock=CAGR_stock,
                           CAGR_benchmark=CAGR_benchmark,
                           volatility=volatility,
                           sharpe=sharpe,
                           correlation=correlation,
                           alpha=alpha,
                           sortino=sortino,
                           maximum_drawdown=maximum_drawdown,
                           calmar=calmar,
                           VaR_90=VaR_90,
                           VaR_95=VaR_95,
                           VaR_99=VaR_99)


@app.route("/about_me")
def about():
    return render_template('about_me.html', title='About')


if __name__ == '__main__':
    app.run(debug=True)