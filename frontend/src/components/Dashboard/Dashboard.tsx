import { useEffect, useState } from "react";

export default function ForecastCard() {
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string>("");

  useEffect(() => {
    fetch("/api/forecast/ets?horizon=6")  // or /arima?sku=Paracetamol
      .then(r => r.json())
      .then(setData)
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="text-red-600">{err}</div>;
  if (!data) return <div>Loading forecast…</div>;

  const first = data.results?.[0];
  return (
    <div>
      <h3 className="font-semibold">ETS Forecast (next {data.horizon} months)</h3>
      <p>Item: {first?.item}</p>
      <p>MAE: {first?.metrics?.MAE?.toFixed(2)} | MSE: {first?.metrics?.MSE?.toFixed(2)} | MAPE: {first?.metrics?.MAPE?.toFixed(2)}%</p>
      <ul className="list-disc pl-6">
        {first?.future_index?.map((d: string, i: number) => (
          <li key={d}>{d}: {first.future_forecast[i].toFixed(2)}</li>
        ))}
      </ul>
    </div>
  );
}
