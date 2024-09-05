def guarantee_minimum_dimenions(min_x, min_y, max_x, max_y, min_dim_w=None, min_dim_h=None):
    if min_dim_w:
        width = max_x - min_x
        
        inc_x = ((min_dim_w - width) // 2) if (min_dim_w - width // 2) > 0 else 0

        min_x -= inc_x
        max_x += inc_x
        
        max_x = max_x + 1 if (min_dim_w - width)%2 == 1 else max_x
        
    if min_dim_h:        
        height = max_y - min_y
        
        inc_y = ((min_dim_h - height) // 2) if (min_dim_h - height // 2) > 0 else 0
        
        min_y -= inc_y
        max_y += inc_y
        
        max_y = max_y + 1 if (min_dim_h - height)%2 == 1 else max_y

    return min_x, min_y, max_x, max_y