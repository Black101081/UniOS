//+------------------------------------------------------------------+
//|                                          XAUUSD_Scalper_M1.mq5 |
//|                                                           UniOS |
//|                                                                  |
//| Strategy : EMA Crossover + RSI Filter                            |
//| Symbol   : XAUUSD (Gold)                                         |
//| Timeframe: M1 (Scalping)                                         |
//|                                                                  |
//| Entry Logic:                                                     |
//|   BUY  - Fast EMA crosses above Slow EMA, RSI < Overbought      |
//|   SELL - Fast EMA crosses below Slow EMA, RSI > Oversold        |
//|                                                                  |
//| Risk Management:                                                 |
//|   - ATR-based dynamic Stop Loss & Take Profit                   |
//|   - Fixed % risk per trade                                       |
//|   - Spread filter to avoid high-cost entries                    |
//|   - Session time filter                                          |
//+------------------------------------------------------------------+
#property copyright "UniOS"
#property link      ""
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

//--- Input Parameters
input group "=== ENTRY SETTINGS ==="
input int    FastEMA_Period    = 8;      // Fast EMA Period
input int    SlowEMA_Period    = 21;     // Slow EMA Period
input int    RSI_Period        = 14;     // RSI Period
input double RSI_Overbought    = 70.0;  // RSI Overbought Level
input double RSI_Oversold      = 30.0;  // RSI Oversold Level

input group "=== RISK MANAGEMENT ==="
input double RiskPercent       = 1.0;   // Risk % per trade (of balance)
input double ATR_Multiplier_SL = 1.5;   // ATR Multiplier for Stop Loss
input double ATR_Multiplier_TP = 2.5;   // ATR Multiplier for Take Profit
input int    ATR_Period        = 14;    // ATR Period
input int    MaxTrades         = 1;     // Max concurrent open trades

input group "=== FILTER SETTINGS ==="
input int    MaxSpread         = 30;    // Max allowed spread (points)
input int    TradingHourStart  = 8;     // Session start hour (server time)
input int    TradingHourEnd    = 20;    // Session end hour (server time)

input group "=== EA SETTINGS ==="
input long   MagicNumber       = 20260314; // EA Magic Number
input string TradeComment      = "XAUUSD_Scalper_M1"; // Trade comment

//--- Global handles
int g_emaFastHandle  = INVALID_HANDLE;
int g_emaSlowHandle  = INVALID_HANDLE;
int g_rsiHandle      = INVALID_HANDLE;
int g_atrHandle      = INVALID_HANDLE;

CTrade g_trade;

//+------------------------------------------------------------------+
//| Expert initialization                                            |
//+------------------------------------------------------------------+
int OnInit()
{
    //--- Validate inputs
    if(FastEMA_Period >= SlowEMA_Period)
    {
        Alert("FastEMA_Period must be less than SlowEMA_Period!");
        return(INIT_PARAMETERS_INCORRECT);
    }
    if(RiskPercent <= 0 || RiskPercent > 10)
    {
        Alert("RiskPercent must be between 0.1 and 10!");
        return(INIT_PARAMETERS_INCORRECT);
    }

    //--- Create indicator handles
    g_emaFastHandle = iMA(_Symbol, PERIOD_M1, FastEMA_Period, 0, MODE_EMA, PRICE_CLOSE);
    g_emaSlowHandle = iMA(_Symbol, PERIOD_M1, SlowEMA_Period, 0, MODE_EMA, PRICE_CLOSE);
    g_rsiHandle     = iRSI(_Symbol, PERIOD_M1, RSI_Period, PRICE_CLOSE);
    g_atrHandle     = iATR(_Symbol, PERIOD_M1, ATR_Period);

    if(g_emaFastHandle == INVALID_HANDLE ||
       g_emaSlowHandle == INVALID_HANDLE ||
       g_rsiHandle     == INVALID_HANDLE ||
       g_atrHandle     == INVALID_HANDLE)
    {
        Print("ERROR: Failed to create indicator handles. Error: ", GetLastError());
        return(INIT_FAILED);
    }

    //--- Configure trade object
    g_trade.SetMagicNumber(MagicNumber);
    g_trade.SetDeviationInPoints(10);
    g_trade.SetTypeFilling(ORDER_FILLING_IOC);

    Print("XAUUSD Scalper M1 initialized | Magic: ", MagicNumber,
          " | EMA(", FastEMA_Period, "/", SlowEMA_Period, ")",
          " | RSI(", RSI_Period, ")",
          " | ATR(", ATR_Period, ")");

    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    if(g_emaFastHandle != INVALID_HANDLE) IndicatorRelease(g_emaFastHandle);
    if(g_emaSlowHandle != INVALID_HANDLE) IndicatorRelease(g_emaSlowHandle);
    if(g_rsiHandle     != INVALID_HANDLE) IndicatorRelease(g_rsiHandle);
    if(g_atrHandle     != INVALID_HANDLE) IndicatorRelease(g_atrHandle);

    Print("XAUUSD Scalper M1 deinitialized. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick handler                                              |
//+------------------------------------------------------------------+
void OnTick()
{
    //--- Only process on new M1 bar (bar-close strategy)
    static datetime s_lastBarTime = 0;
    datetime currentBarTime = iTime(_Symbol, PERIOD_M1, 0);
    if(currentBarTime == s_lastBarTime) return;
    s_lastBarTime = currentBarTime;

    //--- Session time filter
    if(!IsWithinTradingSession()) return;

    //--- Spread filter
    long currentSpread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
    if(currentSpread > MaxSpread)
    {
        PrintFormat("Spread too high: %d pts (max %d). Skipping.", currentSpread, MaxSpread);
        return;
    }

    //--- Max trades filter
    if(CountOpenTrades() >= MaxTrades) return;

    //--- Read indicator buffers (need 3 bars: [0]=current, [1]=prev closed, [2]=two bars ago)
    double emaFast[3], emaSlow[3], rsi[3], atr[3];
    ArraySetAsSeries(emaFast, true);
    ArraySetAsSeries(emaSlow, true);
    ArraySetAsSeries(rsi,     true);
    ArraySetAsSeries(atr,     true);

    if(CopyBuffer(g_emaFastHandle, 0, 0, 3, emaFast) < 3) return;
    if(CopyBuffer(g_emaSlowHandle, 0, 0, 3, emaSlow) < 3) return;
    if(CopyBuffer(g_rsiHandle,     0, 0, 3, rsi)     < 3) return;
    if(CopyBuffer(g_atrHandle,     0, 0, 3, atr)     < 3) return;

    //--- Use values from fully closed bar [1]
    double fastNow  = emaFast[1];
    double fastPrev = emaFast[2];
    double slowNow  = emaSlow[1];
    double slowPrev = emaSlow[2];
    double rsiNow   = rsi[1];
    double currentATR = atr[1];

    //--- Crossover detection
    bool emaBullCross = (fastPrev <= slowPrev) && (fastNow > slowNow); // Fast crossed above Slow
    bool emaBearCross = (fastPrev >= slowPrev) && (fastNow < slowNow); // Fast crossed below Slow

    //--- Entry signals
    bool buySignal  = emaBullCross && (rsiNow < RSI_Overbought);
    bool sellSignal = emaBearCross && (rsiNow > RSI_Oversold);

    if(!buySignal && !sellSignal) return;

    //--- Calculate SL/TP distances (in price units)
    double slDist = currentATR * ATR_Multiplier_SL;
    double tpDist = currentATR * ATR_Multiplier_TP;

    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double lot = CalculateLotSize(slDist);

    if(lot <= 0)
    {
        Print("ERROR: Invalid lot size calculated.");
        return;
    }

    if(buySignal)
    {
        double sl = NormalizeDouble(ask - slDist, _Digits);
        double tp = NormalizeDouble(ask + tpDist, _Digits);
        PrintFormat("BUY Signal | Lot:%.2f | Ask:%.5f | SL:%.5f | TP:%.5f | ATR:%.5f | RSI:%.2f",
                    lot, ask, sl, tp, currentATR, rsiNow);
        g_trade.Buy(lot, _Symbol, ask, sl, tp, TradeComment);
    }
    else if(sellSignal)
    {
        double sl = NormalizeDouble(bid + slDist, _Digits);
        double tp = NormalizeDouble(bid - tpDist, _Digits);
        PrintFormat("SELL Signal | Lot:%.2f | Bid:%.5f | SL:%.5f | TP:%.5f | ATR:%.5f | RSI:%.2f",
                    lot, bid, sl, tp, currentATR, rsiNow);
        g_trade.Sell(lot, _Symbol, bid, sl, tp, TradeComment);
    }
}

//+------------------------------------------------------------------+
//| Check if current server time is within trading session           |
//+------------------------------------------------------------------+
bool IsWithinTradingSession()
{
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);

    // Skip weekends
    if(dt.day_of_week == 0 || dt.day_of_week == 6) return false;

    return (dt.hour >= TradingHourStart && dt.hour < TradingHourEnd);
}

//+------------------------------------------------------------------+
//| Count open trades matching this EA's symbol and magic number     |
//+------------------------------------------------------------------+
int CountOpenTrades()
{
    int count = 0;
    int total = PositionsTotal();

    for(int i = total - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(ticket == 0) continue;

        if(PositionSelectByTicket(ticket))
        {
            if(PositionGetString(POSITION_SYMBOL)    == _Symbol &&
               PositionGetInteger(POSITION_MAGIC)    == MagicNumber)
            {
                count++;
            }
        }
    }

    return count;
}

//+------------------------------------------------------------------+
//| Calculate lot size based on % risk and ATR stop loss             |
//+------------------------------------------------------------------+
double CalculateLotSize(double slDistance)
{
    if(slDistance <= 0) return 0.01;

    double balance   = AccountInfoDouble(ACCOUNT_BALANCE);
    double riskAmt   = balance * RiskPercent / 100.0;
    double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);

    if(tickValue <= 0 || tickSize <= 0) return 0.01;

    // lotSize = riskAmt / (slDistanceInTicks * tickValue)
    double slInTicks = slDistance / tickSize;
    double lotSize   = riskAmt / (slInTicks * tickValue);

    // Normalize to broker lot constraints
    double minLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double maxLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

    lotSize = MathFloor(lotSize / lotStep) * lotStep;
    lotSize = MathMax(lotSize, minLot);
    lotSize = MathMin(lotSize, maxLot);

    return lotSize;
}
//+------------------------------------------------------------------+
