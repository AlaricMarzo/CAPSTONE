import { useState, useEffect } from "react";
import { Sidebar } from "@/components/Layout/Sidebar";
import { MetricCard } from "@/components/Dashboard/MetricCard";
import { RecentAlertsCard } from "@/components/Dashboard/RecentAlertCard";
import { SalesTrendsChart } from "@/components/Dashboard/SalesTrendCard";
import { ProductTrafficCard } from "@/components/Dashboard/ProductTrafficCard";
import {
  TrendingUp,
  DollarSign,
  Package,
  ShoppingCart,
  User,
  Search,
  Calendar,
  FileText
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { getUserMetrics } from "@/data/mockData";
import DataUploadPage from "./upload";
import ProfilePage from "./Profile";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import ProgressBar from "@/components/ui/ProgressBar";

function HomePage({ onProfileClick, onLogout }: { onProfileClick: () => void; onLogout: () => void }) {
  const metrics = getUserMetrics();
  const [confirmOpen, setConfirmOpen] = useState(false);

  const confirmLogout = () => {
    onLogout();
    setConfirmOpen(false);
  };

  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Dashboard Overview</h2>
          <p className="text-muted-foreground">Real-time business insights and analytics</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Calendar className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">Last 30 Days</span>
          </div>
          <Button variant="outline" size="sm" className="shadow-soft">
            <FileText className="h-4 w-4 mr-2" />
            Export Report
          </Button>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                <User className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={onProfileClick}>
                <User className="mr-2 h-4 w-4" />
                Profile
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                className="text-destructive cursor-pointer"
                onSelect={(e) => {
                  e.preventDefault();
                  setConfirmOpen(true);
                }}
              >
                Log out
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          <AlertDialog open={confirmOpen} onOpenChange={setConfirmOpen}>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Are you sure you want to log out?</AlertDialogTitle>
                <AlertDialogDescription>You will be returned to the login page.</AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction onClick={confirmLogout}>Yes, log me out</AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
          <Input placeholder="Search products, customers, orders..." className="pl-10 shadow-soft" />
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard title="Total Revenue" value={`$${metrics.totalRevenue.toFixed(2)}`} change="+12.5% from last month" changeType="positive" icon={DollarSign} color="success" className="shadow-soft hover:shadow-elegant transition-shadow" />
        <MetricCard title="Total Sales" value={metrics.totalSales.toString()} change="+8.2% from last month" changeType="positive" icon={ShoppingCart} color="info" className="shadow-soft hover:shadow-elegant transition-shadow" />
        <MetricCard title="Low Stock Items" value={metrics.lowStockItems.toString()} change={`${metrics.outOfStockItems} out of stock`} changeType="warning" icon={Package} color="warning" className="shadow-soft hover:shadow-elegant transition-shadow" />
        <MetricCard title="Avg Order Value" value={`$${metrics.averageOrderValue.toFixed(2)}`} change="+5.1% from last month" changeType="positive" icon={TrendingUp} color="default" className="shadow-soft hover:shadow-elegant transition-shadow" />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2"><SalesTrendsChart /></div>
        <ProductTrafficCard />
      </div>

      <RecentAlertsCard />
    </div>
  );
}

function SalesPage() {
  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      <div>
        <h2 className="text-3xl font-bold text-foreground">Sales Dashboard</h2>
        <p className="text-muted-foreground">Sales Trends, Forecasting & Product Performance</p>
      </div>
      <div className="text-center py-12 text-muted-foreground">
        <p>Sales dashboard content coming soon...</p>
      </div>
    </div>
  );
}

function InventoryPage() {
  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      <div>
        <h2 className="text-3xl font-bold text-foreground">Inventory</h2>
        <p className="text-muted-foreground">Live Stock Monitoring & Allocation Logic</p>
      </div>
      <div className="text-center py-12 text-muted-foreground">
        <p>Inventory management content coming soon...</p>
      </div>
    </div>
  );
}

function DescriptivePage({ data, loading, error, runningAnalysis, onRunAnalysis }: {
  data: any;
  loading: boolean;
  error: string | null;
  runningAnalysis: boolean;
  onRunAnalysis: () => void;
}) {
  if (loading) {
    return (
      <div className="flex-1 space-y-6 p-8 pt-6">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Descriptive Analytics</h2>
          <p className="text-muted-foreground">KPIs, Market Basket Analysis & Clustering</p>
        </div>
        <div className="text-center py-12 text-muted-foreground">
          <p>Loading analytics data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex-1 space-y-6 p-8 pt-6">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Descriptive Analytics</h2>
          <p className="text-muted-foreground">KPIs, Market Basket Analysis & Clustering</p>
        </div>
        <div className="text-center py-12 text-destructive">
          <p>Error loading data: {error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Descriptive Analytics</h2>
          <p className="text-muted-foreground">KPIs, Market Basket Analysis & Clustering</p>
        </div>
        <Button
          onClick={onRunAnalysis}
          disabled={runningAnalysis}
          className="shadow-soft"
        >
          {runningAnalysis ? "Running Analysis..." : "Run Descriptive Analysis"}
        </Button>
      </div>

      {runningAnalysis && (
        <ProgressBar label="Running Descriptive Analysis..." />
      )}

      <Tabs defaultValue="kpis" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="kpis">KPIs</TabsTrigger>
          <TabsTrigger value="mba">Market Basket Analysis</TabsTrigger>
          <TabsTrigger value="clustering">Clustering</TabsTrigger>
        </TabsList>

        <TabsContent value="kpis" className="space-y-4">
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data?.kpis}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="sales" stroke="#8884d8" name="Sales" />
                <Line type="monotone" dataKey="revenue" stroke="#82ca9d" name="Revenue" />
                <Line type="monotone" dataKey="profit" stroke="#ffc658" name="Profit" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </TabsContent>

        <TabsContent value="mba" className="space-y-4">
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data?.mbaRules?.slice(0, 10)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="antecedents" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="support" fill="#8884d8" name="Support" />
                <Bar dataKey="confidence" fill="#82ca9d" name="Confidence" />
                <Bar dataKey="lift" fill="#ffc658" name="Lift" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </TabsContent>

        <TabsContent value="clustering" className="space-y-4">
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="feature_1" name="Feature 1" />
                <YAxis dataKey="feature_2" name="Feature 2" />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Legend />
                <Scatter name="Clusters" data={data?.clustering} fill="#8884d8" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}

function PredictivePage({ data, loading, error, runningAnalysis, onRunAnalysis }: {
  data: any;
  loading: boolean;
  error: string | null;
  runningAnalysis: boolean;
  onRunAnalysis: () => void;
}) {
  if (loading) {
    return (
      <div className="flex-1 space-y-6 p-8 pt-6">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Predictive Analytics</h2>
          <p className="text-muted-foreground">Forecasting, Machine Learning Models & Time Series Analysis</p>
        </div>
        <div className="text-center py-12 text-muted-foreground">
          <p>Loading predictive analytics data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex-1 space-y-6 p-8 pt-6">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Predictive Analytics</h2>
          <p className="text-muted-foreground">Forecasting, Machine Learning Models & Time Series Analysis</p>
        </div>
        <div className="text-center py-12 text-destructive">
          <p>Error loading data: {error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Predictive Analytics</h2>
          <p className="text-muted-foreground">Forecasting, Machine Learning Models & Time Series Analysis</p>
        </div>
        <Button
          onClick={onRunAnalysis}
          disabled={runningAnalysis}
          className="shadow-soft"
        >
          {runningAnalysis ? "Running Analysis..." : "Run Predictive Analysis"}
        </Button>
      </div>

      {runningAnalysis && (
        <ProgressBar label="Running Predictive Analysis..." />
      )}

      <Tabs defaultValue="random-forest" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="random-forest">Random Forest</TabsTrigger>
          <TabsTrigger value="xgboost">XGBoost</TabsTrigger>
          <TabsTrigger value="sarima">SARIMA/ETS</TabsTrigger>
        </TabsList>

        <TabsContent value="random-forest" className="space-y-4">
          <Tabs defaultValue="forecasts" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="forecasts">Forecasts</TabsTrigger>
              <TabsTrigger value="features">Feature Importance</TabsTrigger>
            </TabsList>

            <TabsContent value="forecasts" className="space-y-4">
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={data?.randomForest?.forecasts}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="actual" stroke="#8884d8" name="Actual" />
                    <Line type="monotone" dataKey="predicted" stroke="#82ca9d" name="Predicted" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>

            <TabsContent value="features" className="space-y-4">
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data?.randomForest?.features} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis dataKey="feature" type="category" />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="importance" fill="#8884d8" name="Importance" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>
          </Tabs>
        </TabsContent>

        <TabsContent value="xgboost" className="space-y-4">
          <Tabs defaultValue="forecasts" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="forecasts">Forecasts</TabsTrigger>
              <TabsTrigger value="features">Feature Importance</TabsTrigger>
            </TabsList>

            <TabsContent value="forecasts" className="space-y-4">
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={data?.xgboost?.forecasts}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="actual" stroke="#8884d8" name="Actual" />
                    <Line type="monotone" dataKey="predicted" stroke="#82ca9d" name="Predicted" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>

            <TabsContent value="features" className="space-y-4">
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data?.xgboost?.features} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis dataKey="feature" type="category" />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="importance" fill="#8884d8" name="Importance" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>
          </Tabs>
        </TabsContent>

        <TabsContent value="sarima" className="space-y-4">
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data?.sarima?.forecasts}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="actual" stroke="#8884d8" name="Actual" />
                <Line type="monotone" dataKey="predicted" stroke="#82ca9d" name="Predicted" />
                <Line type="monotone" dataKey="lower_bound" stroke="#ffc658" name="Lower Bound" strokeDasharray="3 3" />
                <Line type="monotone" dataKey="upper_bound" stroke="#ff8042" name="Upper Bound" strokeDasharray="3 3" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}

function PrescriptivePage({ data, loading, error, runningAnalysis, onRunAnalysis }: {
  data: any;
  loading: boolean;
  error: string | null;
  runningAnalysis: boolean;
  onRunAnalysis: () => void;
}) {
  if (loading) {
    return (
      <div className="flex-1 space-y-6 p-8 pt-6">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Prescriptive Analytics</h2>
          <p className="text-muted-foreground">Optimization, Recommendations & Decision Support</p>
        </div>
        <div className="text-center py-12 text-muted-foreground">
          <p>Loading prescriptive analytics data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex-1 space-y-6 p-8 pt-6">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Prescriptive Analytics</h2>
          <p className="text-muted-foreground">Optimization, Recommendations & Decision Support</p>
        </div>
        <div className="text-center py-12 text-destructive">
          <p>Error loading data: {error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Prescriptive Analytics</h2>
          <p className="text-muted-foreground">Optimization, Recommendations & Decision Support</p>
        </div>
        <Button
          onClick={onRunAnalysis}
          disabled={runningAnalysis}
          className="shadow-soft"
        >
          {runningAnalysis ? "Running Analysis..." : "Run Prescriptive Analysis"}
        </Button>
      </div>

      {runningAnalysis && (
        <ProgressBar label="Running Prescriptive Analysis..." />
      )}

      <Tabs defaultValue="reorder-point" className="w-full">
        <TabsList className="grid w-full grid-cols-8">
          <TabsTrigger value="reorder-point">Reorder Point</TabsTrigger>
          <TabsTrigger value="eoq">EOQ</TabsTrigger>
          <TabsTrigger value="inventory-allocation">Inventory Allocation</TabsTrigger>
          <TabsTrigger value="what-if">What-If Analysis</TabsTrigger>
          <TabsTrigger value="discount-optimization">Discount Optimization</TabsTrigger>
          <TabsTrigger value="resource-planning">Resource Planning</TabsTrigger>
          <TabsTrigger value="anomaly-detection">Anomaly Detection</TabsTrigger>
          <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
        </TabsList>

        <TabsContent value="reorder-point" className="space-y-4">
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data?.reorderPoint}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="medicine" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="reorder_point" fill="#8884d8" name="Reorder Point" />
                <Bar dataKey="safety_stock" fill="#82ca9d" name="Safety Stock" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </TabsContent>

        <TabsContent value="eoq" className="space-y-4">
          <Tabs defaultValue="eoq-values" className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="eoq-values">EOQ Values</TabsTrigger>
              <TabsTrigger value="order-frequency">Order Frequency</TabsTrigger>
              <TabsTrigger value="cost-breakdown">Cost Breakdown</TabsTrigger>
              <TabsTrigger value="efficiency">Efficiency</TabsTrigger>
            </TabsList>

            <TabsContent value="eoq-values" className="space-y-4">
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data?.eoq}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="medicine" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="eoq" fill="#8884d8" name="EOQ (Units)" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>

            <TabsContent value="order-frequency" className="space-y-4">
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={data?.eoq}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="medicine" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="orders_per_year" stroke="#8884d8" name="Orders per Year" />
                    <Line type="monotone" dataKey="days_between_orders" stroke="#82ca9d" name="Days Between Orders" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>

            <TabsContent value="cost-breakdown" className="space-y-4">
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data?.eoq}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="medicine" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="annual_ordering_cost" stackId="a" fill="#8884d8" name="Ordering Cost" />
                    <Bar dataKey="annual_holding_cost" stackId="a" fill="#82ca9d" name="Holding Cost" />
                    <Bar dataKey="annual_purchase_cost" stackId="a" fill="#ffc658" name="Purchase Cost" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>

            <TabsContent value="efficiency" className="space-y-4">
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart data={data?.eoq}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="annual_demand" name="Annual Demand" />
                    <YAxis dataKey="total_annual_cost" name="Total Annual Cost" />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                    <Legend />
                    <Scatter name="EOQ Efficiency" dataKey="total_annual_cost" fill="#8884d8" />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>
          </Tabs>
        </TabsContent>

        <TabsContent value="inventory-allocation" className="space-y-4">
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data?.inventoryAllocation}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="medicine" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="optimal_allocation" fill="#8884d8" name="Allocated Quantity" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </TabsContent>

        <TabsContent value="what-if" className="space-y-4">
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data?.whatIfAnalysis}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="scenario" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="projected_profit" stroke="#8884d8" name="Profit" />
                <Line type="monotone" dataKey="projected_cost" stroke="#82ca9d" name="Cost" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </TabsContent>

        <TabsContent value="discount-optimization" className="space-y-4">
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart data={data?.discountByProduct}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="discount_pct" name="Discount %" />
                <YAxis dataKey="profit_margin" name="Profit Margin %" />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Legend />
                <Scatter name="Products" dataKey="profit_margin" fill="#8884d8" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </TabsContent>

        <TabsContent value="resource-planning" className="space-y-4">
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data?.resourcePlanning}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="medicine" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="storage_needed" fill="#8884d8" name="Storage Needed" />
                <Bar dataKey="capital_needed" fill="#82ca9d" name="Capital Needed" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </TabsContent>

        <TabsContent value="anomaly-detection" className="space-y-4">
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart data={data?.anomalyDetection}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" name="Date" />
                <YAxis dataKey="sales" name="Sales" />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Legend />
                <Scatter data={data?.anomalyDetection?.filter((d: any) => d.anomaly === 1)} name="Normal" fill="#82ca9d" />
                <Scatter data={data?.anomalyDetection?.filter((d: any) => d.anomaly === -1)} name="Anomaly" fill="#ff7300" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </TabsContent>

        <TabsContent value="recommendations" className="space-y-4">
          <div className="max-h-96 overflow-y-auto p-4 bg-muted rounded-lg">
            <pre className="whitespace-pre-wrap text-sm">{data?.recommendations}</pre>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}

// Create a Dashboard component that combines all the pages
interface DashboardProps {
  onLogout: () => void;
  userEmail: string;
}

export default function Dashboard({ onLogout, userEmail }: DashboardProps) {
  const [activeTab, setActiveTab] = useState("home");
  const [showProfile, setShowProfile] = useState(false);

  // Shared state for analytics data
  const [descriptiveData, setDescriptiveData] = useState<any>(null);
  const [descriptiveLoading, setDescriptiveLoading] = useState(true);
  const [descriptiveError, setDescriptiveError] = useState<string | null>(null);
  const [descriptiveRunning, setDescriptiveRunning] = useState(false);

  const [predictiveData, setPredictiveData] = useState<any>(null);
  const [predictiveLoading, setPredictiveLoading] = useState(true);
  const [predictiveError, setPredictiveError] = useState<string | null>(null);
  const [predictiveRunning, setPredictiveRunning] = useState(false);

  const [prescriptiveData, setPrescriptiveData] = useState<any>(null);
  const [prescriptiveLoading, setPrescriptiveLoading] = useState(true);
  const [prescriptiveError, setPrescriptiveError] = useState<string | null>(null);
  const [prescriptiveRunning, setPrescriptiveRunning] = useState(false);

  const handleProfileClick = () => {
    setShowProfile(true);
    setActiveTab("profile");
  };

  const handleBackToDashboard = () => {
    setShowProfile(false);
    setActiveTab("home");
  };

  // Fetch functions
  const fetchDescriptiveData = async () => {
    try {
      setDescriptiveLoading(true);
      const response = await fetch('/api/analytics/descriptive');
      if (!response.ok) {
        throw new Error('Failed to fetch data');
      }
      const result = await response.json();
      setDescriptiveData(result.data);
      setDescriptiveError(null);
    } catch (err) {
      setDescriptiveError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setDescriptiveLoading(false);
    }
  };

  const fetchPredictiveData = async () => {
    try {
      setPredictiveLoading(true);
      const response = await fetch('/api/analytics/predictive');
      if (!response.ok) {
        throw new Error('Failed to fetch data');
      }
      const result = await response.json();
      setPredictiveData(result.data);
      setPredictiveError(null);
    } catch (err) {
      setPredictiveError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setPredictiveLoading(false);
    }
  };

  const fetchPrescriptiveData = async () => {
    try {
      setPrescriptiveLoading(true);
      const response = await fetch('/api/analytics/prescriptive');
      if (!response.ok) {
        throw new Error('Failed to fetch data');
      }
      const result = await response.json();
      setPrescriptiveData(result.data);
      setPrescriptiveError(null);
    } catch (err) {
      setPrescriptiveError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setPrescriptiveLoading(false);
    }
  };

  // Run analysis functions
  const runDescriptiveAnalysis = async () => {
    try {
      setDescriptiveRunning(true);
      const response = await fetch('/api/analytics/run-descriptive', {
        method: 'POST',
      });
      if (!response.ok) {
        throw new Error('Failed to run analysis');
      }
      const result = await response.json();
      if (result.success) {
        await fetchDescriptiveData();
      } else {
        setDescriptiveError(result.error || 'Analysis failed');
      }
    } catch (err) {
      setDescriptiveError(err instanceof Error ? err.message : 'Failed to run analysis');
    } finally {
      setDescriptiveRunning(false);
    }
  };

  const runPredictiveAnalysis = async () => {
    try {
      setPredictiveRunning(true);
      const response = await fetch('/api/analytics/run-predictive', {
        method: 'POST',
      });
      if (!response.ok) {
        throw new Error('Failed to run analysis');
      }
      const result = await response.json();
      if (result.success) {
        await fetchPredictiveData();
      } else {
        setPredictiveError(result.error || 'Analysis failed');
      }
    } catch (err) {
      setPredictiveError(err instanceof Error ? err.message : 'Failed to run analysis');
    } finally {
      setPredictiveRunning(false);
    }
  };

  const runPrescriptiveAnalysis = async () => {
    try {
      setPrescriptiveRunning(true);
      const response = await fetch('/api/analytics/run-prescriptive', {
        method: 'POST',
      });
      if (!response.ok) {
        throw new Error('Failed to run analysis');
      }
      const result = await response.json();
      if (result.success) {
        await fetchPrescriptiveData();
      } else {
        setPrescriptiveError(result.error || 'Analysis failed');
      }
    } catch (err) {
      setPrescriptiveError(err instanceof Error ? err.message : 'Failed to run analysis');
    } finally {
      setPrescriptiveRunning(false);
    }
  };

  // Initial data fetch
  useEffect(() => {
    fetchDescriptiveData();
    fetchPredictiveData();
    fetchPrescriptiveData();
  }, []);

  // Event listener for analytics updates
  useEffect(() => {
    const handleUpdate = () => {
      fetchDescriptiveData();
      fetchPredictiveData();
      fetchPrescriptiveData();
    };
    window.addEventListener('analyticsUpdated', handleUpdate);
    return () => window.removeEventListener('analyticsUpdated', handleUpdate);
  }, []);

  return (
    <div className="flex h-screen bg-background">
      <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />
      <div className="flex-1 flex flex-col overflow-hidden">
        <main className="flex-1 overflow-y-auto">
          {activeTab === "home" && <HomePage onProfileClick={handleProfileClick} onLogout={onLogout} />}
          {activeTab === "sales" && <SalesPage />}
          {activeTab === "inventory" && <InventoryPage />}
          {activeTab === "upload" && <DataUploadPage />}
          {activeTab === "descriptive" && (
            <DescriptivePage
              data={descriptiveData}
              loading={descriptiveLoading}
              error={descriptiveError}
              runningAnalysis={descriptiveRunning}
              onRunAnalysis={runDescriptiveAnalysis}
            />
          )}
          {activeTab === "predictive" && (
            <PredictivePage
              data={predictiveData}
              loading={predictiveLoading}
              error={predictiveError}
              runningAnalysis={predictiveRunning}
              onRunAnalysis={runPredictiveAnalysis}
            />
          )}
          {activeTab === "prescriptive" && (
            <PrescriptivePage
              data={prescriptiveData}
              loading={prescriptiveLoading}
              error={prescriptiveError}
              runningAnalysis={prescriptiveRunning}
              onRunAnalysis={runPrescriptiveAnalysis}
            />
          )}
          {activeTab === "profile" && <ProfilePage userEmail={userEmail} onBack={handleBackToDashboard} onLogout={onLogout} />}
        </main>
      </div>
    </div>
  );
}
