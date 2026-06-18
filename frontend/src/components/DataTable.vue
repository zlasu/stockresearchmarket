<template>
  <div class="table-wrap">
    <table class="data-table">
      <thead>
        <tr v-for="headerGroup in table.getHeaderGroups()" :key="headerGroup.id">
          <th v-for="header in headerGroup.headers" :key="header.id">
            <FlexRender :render="header.column.columnDef.header" :props="header.getContext()" />
          </th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in table.getRowModel().rows" :key="row.id">
          <td v-for="cell in row.getVisibleCells()" :key="cell.id">
            <FlexRender :render="cell.column.columnDef.cell" :props="cell.getContext()" />
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { FlexRender, createColumnHelper, getCoreRowModel, useVueTable } from "@tanstack/vue-table";

import { formatMetric } from "../lib/format";
import type { MetricValue } from "../types";

const props = defineProps<{
  rows: Array<Record<string, MetricValue>>;
  keys: string[];
}>();

const columnHelper = createColumnHelper<Record<string, MetricValue>>();
const columns = computed(() =>
  props.keys.map((key) =>
    columnHelper.accessor((row) => row[key], {
      id: key,
      header: () => key.replaceAll("_", " "),
      cell: (info) => formatMetric(key, info.getValue()),
    }),
  ),
);

const table = useVueTable({
  get data() {
    return props.rows;
  },
  get columns() {
    return columns.value;
  },
  getCoreRowModel: getCoreRowModel(),
});
</script>

