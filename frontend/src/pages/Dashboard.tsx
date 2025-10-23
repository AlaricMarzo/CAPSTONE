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
import UploadPage from "./upload";
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

function ReportsPage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [runningAnalysis, setRunningAnalysis] = useState(false);

  const fetchData = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/analytics/prescriptive');
      if (!response.ok) {
        throw new Error('Failed to fetch data');
      }
      const result = await response.json();
      setData(result.data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const runAnalysis = async () => {
    try {
      setRunningAnalysis(true);
      const response = await fetch('/api/analytics/run-prescriptive', {
        method: 'POST',
      });
      if (!response.ok) {
        throw new Error('Failed to run analysis');
      }
      const result = await response.json();
      if (result.success) {
        // Refresh data after successful run
        await fetchData();
      } else {
        setError(result.error || 'Analysis failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to run analysis');
    } finally {
      setRunningAnalysis(false);
    }
  };

  if (loading) {
    return (
      <div className="flex-1 space-y-6 p-8 pt-6">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Reports</h2>
          <p className="text-muted-foreground">Analytics, Insights & Performance Reports</p>
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
          <h2 className="text-3xl font-bold text-foreground">Reports</h2>
          <p className="text-muted-foreground">Analytics, Insights & Performance Reports</p>
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
          <h2 className="text-3xl font-bold text-foreground">Reports</h2>
          <p className="text-muted-foreground">Analytics, Insights & Performance Reports</p>
        </div>
        <Button
          onClick={runAnalysis}
          disabled={runningAnalysis}
          className="shadow-soft"
        >
          {runningAnalysis ? "Running Analysis..." : "Run Prescriptive Analysis"}
        </Button>
      </div>

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

export default function Dashboard({ onLogout, userEmail }: { onLogout: () => void; userEmail: string }) {
  const [activeTab, setActiveTab] = useState("home");
  const [previousTab, setPreviousTab] = useState("home");

  const handleProfileClick = () => {
    setPreviousTab(activeTab);
    setActiveTab("profile");
  };

  const handleLogout = () => onLogout();

  const renderContent = () => {
    switch (activeTab) {
      case "home":      return <HomePage onProfileClick={handleProfileClick} onLogout={handleLogout} />;
      case "sales":     return <SalesPage />;
      case "inventory": return <InventoryPage />;
      case "reports":   return <ReportsPage />;
      case "upload":    return <UploadPage />;
      case "profile":   return <ProfilePage userEmail={userEmail} onLogout={handleLogout} onBack={() => setActiveTab(previousTab)} />;
      default:          return <HomePage onProfileClick={handleProfileClick} onLogout={handleLogout} />;
    }
  };

  return (
    <div className="flex h-screen bg-gradient-to-br from-background via-background to-muted/20">
      <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />
      <div className="flex-1 flex flex-col overflow-hidden">
        <main className="flex-1 overflow-y-auto">{renderContent()}</main>
      </div>
    </div>
  );
}
