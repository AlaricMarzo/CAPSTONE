import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { LucideIcon } from "lucide-react";

interface MetricCardProps {
  title: string;
  value: string;
  change?: string;
  changeType?: 'positive' | 'negative' | 'neutral' | 'warning';
  icon?: LucideIcon;
  color?: 'default' | 'success' | 'warning' | 'info' | 'destructive';
  className?: string;
}

const colorVariants = {
  default: 'text-primary',
  success: 'text-success',
  warning: 'text-warning',
  info: 'text-info',
  destructive: 'text-destructive'
};

export function MetricCard({ 
  title, 
  value, 
  change, 
  changeType = 'neutral',
  icon: Icon,
  color = 'default',
  className
}: MetricCardProps) {
  return (
    <Card className={cn("relative overflow-hidden", className)}>
      <CardContent className="p-6">
        <div className="flex items-center justify-between">
          <div className="space-y-2">
            <p className="text-sm font-medium text-muted-foreground">{title}</p>
            <p className={cn("text-3xl font-bold", colorVariants[color])}>
              {value}
            </p>
            {change && (
              <p className={cn(
                "text-sm font-medium",
                changeType === 'positive' && "text-success",
                changeType === 'negative' && "text-destructive",
                changeType === 'warning' && "text-warning",
                changeType === 'neutral' && "text-muted-foreground"
              )}>
                {change}
              </p>
            )}
          </div>
          {Icon && (
            <div className={cn(
              "h-12 w-12 rounded-lg flex items-center justify-center",
              color === 'success' && "bg-success/10",
              color === 'warning' && "bg-warning/10", 
              color === 'info' && "bg-info/10",
              color === 'destructive' && "bg-destructive/10",
              color === 'default' && "bg-primary/10"
            )}>
              <Icon className={cn("h-6 w-6", colorVariants[color])} />
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}