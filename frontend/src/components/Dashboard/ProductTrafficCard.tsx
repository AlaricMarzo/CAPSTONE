import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const products = [
  { name: 'Paracetamol Medicine', traffic: 'high', color: 'bg-primary' },
  { name: 'Canned Goods', traffic: 'high', color: 'bg-primary' },
  { name: 'Soy', traffic: 'medium', color: 'bg-success' },
  { name: 'Medicine', traffic: 'low', color: 'bg-info' },
  { name: 'FMCG', traffic: 'low', color: 'bg-info' },
  { name: 'Shirt/x', traffic: 'medium', color: 'bg-success' },
  { name: 'Water', traffic: 'low', color: 'bg-info' },
  { name: 'SoktA', traffic: 'medium', color: 'bg-success' },
];

const trafficLabels = [
  { label: 'Low Traffic', count: '4' },
  { label: 'High Traffic', count: '2' },
];

export function ProductTrafficCard() {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg font-semibold">Product Traffic</CardTitle>
        <p className="text-sm text-muted-foreground">Most viewed or accessed items</p>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-4 gap-2 mb-6">
          {products.map((product, index) => (
            <div
              key={index}
              className={`${product.color} text-white rounded-lg p-3 text-center text-sm font-medium`}
            >
              {product.name}
            </div>
          ))}
        </div>
        
        <div className="space-y-3">
          {trafficLabels.map((item, index) => (
            <div key={index} className="flex items-center justify-between">
              <span className="text-sm text-foreground">{item.label}</span>
              <span className="text-sm font-medium text-muted-foreground">{item.count}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}