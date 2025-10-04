import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle, TrendingUp, Package, Activity } from "lucide-react";

const alerts = [
  {
    id: 1,
    type: 'critical',
    title: 'Item out of stock',
    description: 'Paracetamol 500mg',
    time: '32 minutes ago',
    icon: Package,
  },
  {
    id: 2,
    type: 'warning',
    title: 'Unusual Traffic Pattern',
    description: 'Traffic increased on product page',
    time: '45 minutes ago',
    icon: TrendingUp,
  },
  {
    id: 3,
    type: 'info',
    title: 'Inventory Alert',
    description: 'Product ID 68321 is running low (5 units left)',
    time: '2 hours ago',
    icon: AlertTriangle,
  },
  {
    id: 4,
    type: 'success',
    title: 'System Update Complete',
    description: 'Version 2.4.1 deployed successfully',
    time: '3 hours ago',
    icon: Activity,
  },
];

const badgeVariants = {
  critical: 'destructive',
  warning: 'warning',
  info: 'info',
  success: 'success',
} as const;

export function RecentAlertsCard() {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-lg font-semibold">Real-time Alerts</CardTitle>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">System monitoring & anomaly detection</span>
          <Badge variant="destructive" className="text-xs">3 New</Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {alerts.map((alert) => {
            const Icon = alert.icon;
            return (
              <div key={alert.id} className="flex items-start gap-3 p-3 rounded-lg bg-muted/30">
                <div className={`p-2 rounded-lg ${
                  alert.type === 'critical' ? 'bg-destructive/10' :
                  alert.type === 'warning' ? 'bg-warning/10' :
                  alert.type === 'info' ? 'bg-info/10' :
                  'bg-success/10'
                }`}>
                  <Icon className={`h-4 w-4 ${
                    alert.type === 'critical' ? 'text-destructive' :
                    alert.type === 'warning' ? 'text-warning' :
                    alert.type === 'info' ? 'text-info' :
                    'text-success'
                  }`} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <h4 className="text-sm font-medium text-foreground">{alert.title}</h4>
                    <Badge variant={badgeVariants[alert.type]} className="text-xs">
                      {alert.type}
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">{alert.description}</p>
                  <p className="text-xs text-muted-foreground mt-1">{alert.time}</p>
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}