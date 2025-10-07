package main

import "fmt"

func print_wrapper(nums ... interface{}) {
    fmt.Println(nums)
}

func main() {
	print_wrapper(1, 4, 5, "sus", true)
}