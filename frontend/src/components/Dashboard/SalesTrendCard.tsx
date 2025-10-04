import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Line, ComposedChart } from 'recharts';

const data = [
  { week: 'Week 1', sales: 45000, trend: 48000 },
  { week: 'Week 2', sales: 62000, trend: 55000 },
  { week: 'Week 3', sales: 58000, trend: 52000 },
  { week: 'Week 4', sales: 71000, trend: 58000 },
  { week: 'Week 5', sales: 65000, trend: 62000 },
  { week: 'Week 6', sales: 78000, trend: 68000 },
  { week: 'Week 7', sales: 85000, trend: 75000 },
  { week: 'Week 8', sales: 82000, trend: 78000 },
];

export function SalesTrendsChart() {
  return (
    <Card className="col-span-2">
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle className="text-lg font-semibold">Recent Sales Trends</CardTitle>
          <p className="text-sm text-muted-foreground">Weekly monthly performance summary</p>
        </div>
        <div className="flex gap-2">
          <div className="flex items-center gap-2">
            <div className="h-3 w-3 rounded-full bg-primary"></div>
            <span className="text-sm text-muted-foreground">Weekly Sales</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="h-3 w-3 rounded-full bg-muted-foreground"></div>
            <span className="text-sm text-muted-foreground">Trend</span>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis 
                dataKey="week" 
                stroke="hsl(var(--muted-foreground))"
                fontSize={12}
              />
              <YAxis 
                stroke="hsl(var(--muted-foreground))"
                fontSize={12}
                tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
              />
              <Bar 
                dataKey="sales" 
                fill="hsl(var(--primary))" 
                radius={[4, 4, 0, 0]}
                opacity={0.8}
              />
              <Line 
                type="monotone" 
                dataKey="trend" 
                stroke="hsl(var(--muted-foreground))"
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}