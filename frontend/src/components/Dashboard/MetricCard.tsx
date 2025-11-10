import type React from "react"
import { Card, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/utils"

interface MetricCardProps {
  title: string
  value: string | number
  change?: string
  changeType?: "positive" | "negative" | "warning"
  icon?: React.ComponentType<any>
  color?: "success" | "info" | "warning" | "default"
  className?: string
}

export function MetricCard({
  title,
  value,
  change,
  changeType = "positive",
  icon: Icon,
  color = "default",
  className,
}: MetricCardProps) {
  const colorMap = {
    success: "bg-green-50 dark:bg-green-950 text-green-700 dark:text-green-200",
    info: "bg-blue-50 dark:bg-blue-950 text-blue-700 dark:text-blue-200",
    warning: "bg-amber-50 dark:bg-amber-950 text-amber-700 dark:text-amber-200",
    default: "bg-slate-50 dark:bg-slate-900 text-slate-700 dark:text-slate-200",
  }

  const changeColorMap = {
    positive: "text-green-600 dark:text-green-400",
    negative: "text-red-600 dark:text-red-400",
    warning: "text-amber-600 dark:text-amber-400",
  }

  return (
    <Card className={cn("overflow-hidden", className)}>
      <CardContent className="p-6">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <p className="text-sm font-medium text-muted-foreground">{title}</p>
            <h3 className="text-2xl font-bold mt-2">{value}</h3>
            {change && <p className={cn("text-xs mt-2", changeColorMap[changeType])}>{change}</p>}
          </div>
          {Icon && (
            <div className={cn("p-3 rounded-lg", colorMap[color])}>
              <Icon className="w-5 h-5" />
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
