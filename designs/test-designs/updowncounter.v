// Minimal up/down counter: 3 paths (load, up, down), 5 branch points
module updowncounter (
  input clk,
  input load,
  input up,
  input down,
  input [7:0] data_in,
  output reg [7:0] count
);
  always @(posedge clk) begin
    if (load)
      count <= data_in;
    else if (up)
      count <= count + 1;
    else if (down)
      count <= count - 1;
  end
endmodule
