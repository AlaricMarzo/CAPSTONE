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
  
];

const badgeVariants = {
  critical: 'destructive',
  warning: 'warning',
  info: 'info',
  success: 'success',
} as const;
