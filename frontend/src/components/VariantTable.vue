<template>
  <DataTable :rows="rows" :keys="columns" />
</template>

<script setup lang="ts">
import { computed } from "vue";

import DataTable from "./DataTable.vue";
import type { VariantSummary } from "../types";

const props = defineProps<{
  variants: VariantSummary[];
  selected: string[];
}>();

const emit = defineEmits<{
  toggle: [variantId: string];
}>();

const columns = ["selected", "name", "role", "cagr", "sharpe", "max_drawdown", "calmar", "avg_annual_turnover", "trades", "score"];

const rows = computed(() =>
  props.variants.map((variant) => ({
    selected: props.selected.includes(variant.id) ? "selected" : "select",
    name: variant.name,
    role: variant.role ?? variant.kind,
    cagr: variant.metrics.cagr,
    sharpe: variant.metrics.sharpe,
    max_drawdown: variant.metrics.max_drawdown,
    calmar: variant.metrics.calmar,
    avg_annual_turnover: variant.metrics.avg_annual_turnover,
    trades: variant.metrics.trades,
    score: variant.computed_score,
    _id: variant.id,
  })),
);

defineExpose({ emitToggle: (variantId: string) => emit("toggle", variantId) });
</script>

