import { createRouter, createWebHistory } from "vue-router";

import CompareView from "./views/CompareView.vue";
import DashboardView from "./views/DashboardView.vue";
import ExperimentView from "./views/ExperimentView.vue";
import VariantView from "./views/VariantView.vue";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", name: "dashboard", component: DashboardView },
    { path: "/experiments/:id", name: "experiment", component: ExperimentView },
    { path: "/experiments/:id/compare", name: "compare", component: CompareView },
    { path: "/experiments/:id/variants/:variantId", name: "variant", component: VariantView },
  ],
});

export default router;

